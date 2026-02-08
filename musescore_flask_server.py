import math
import os
import pickle
import click
import tempfile
from glob import glob
import subprocess
import zipfile
import traceback
import tempfile

import music21
import numpy as np
from flask import Flask, request, make_response, jsonify
from music21 import musicxml, converter
from tqdm import tqdm
import torch
import logging
from logging import handlers as logging_handlers

from DatasetManager.chorale_dataset import ChoraleDataset
from DatasetManager.dataset_manager import DatasetManager
from DatasetManager.metadata import FermataMetadata, TickMetadata, KeyMetadata
from DeepBach.model_manager import DeepBach

if not zipfile.is_zipfile(file_path):
    return f"Not a zip/mxl or incomplete file: {file_path}", 400

UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = {'xml', 'mxl', 'mid', 'midi'}

app = Flask(__name__)

deepbach = None
_tensor_metadata = None
_num_iterations = None
_sequence_length_ticks = None
_ticks_per_quarter = None
_tensor_sheet = None

# TODO use this parameter or extract it from the metadata somehow
timesignature = music21.meter.TimeSignature('4/4')

# generation parameters
# todo put in click?
batch_size_per_voice = 8

metadatas = [
    FermataMetadata(),
    TickMetadata(subdivision=_ticks_per_quarter),
    KeyMetadata()
]

response_headers = {"Content-Type": "text/html",
                    "charset":      "utf-8"
                    }


@click.command()
@click.option('--note_embedding_dim', default=20,
              help='size of the note embeddings')
@click.option('--meta_embedding_dim', default=20,
              help='size of the metadata embeddings')
@click.option('--num_layers', default=2,
              help='number of layers of the LSTMs')
@click.option('--lstm_hidden_size', default=256,
              help='hidden size of the LSTMs')
@click.option('--dropout_lstm', default=0.5,
              help='amount of dropout between LSTM layers')
@click.option('--dropout_lstm', default=0.5,
              help='amount of dropout between LSTM layers')
@click.option('--linear_hidden_size', default=256,
              help='hidden size of the Linear layers')
@click.option('--num_iterations', default=100,
              help='number of parallel pseudo-Gibbs sampling iterations (for a single update)')
@click.option('--sequence_length_ticks', default=64,
              help='length of the generated chorale (in ticks)')
@click.option('--ticks_per_quarter', default=4,
              help='number of ticks per quarter note')
@click.option('--port', default=5000,
              help='port to serve on')
def init_app(note_embedding_dim,
             meta_embedding_dim,
             num_layers,
             lstm_hidden_size,
             dropout_lstm,
             linear_hidden_size,
             num_iterations,
             sequence_length_ticks,
             ticks_per_quarter,
             port
             ):
    global metadatas
    global _sequence_length_ticks
    global _num_iterations
    global _ticks_per_quarter
    global bach_chorales_dataset

    _ticks_per_quarter = ticks_per_quarter
    _sequence_length_ticks = sequence_length_ticks
    _num_iterations = num_iterations

    dataset_manager = DatasetManager()
    chorale_dataset_kwargs = {
        'voice_ids':      [0, 1, 2, 3],
        'metadatas':      metadatas,
        'sequences_size': 8,
        'subdivision':    4
    }

    _bach_chorales_dataset: ChoraleDataset = dataset_manager.get_dataset(
        name='bach_chorales',
        **chorale_dataset_kwargs
    )
    bach_chorales_dataset = _bach_chorales_dataset

    assert sequence_length_ticks % bach_chorales_dataset.subdivision == 0

    global deepbach
    deepbach = DeepBach(
        dataset=bach_chorales_dataset,
        note_embedding_dim=note_embedding_dim,
        meta_embedding_dim=meta_embedding_dim,
        num_layers=num_layers,
        lstm_hidden_size=lstm_hidden_size,
        dropout_lstm=dropout_lstm,
        linear_hidden_size=linear_hidden_size
    )
    deepbach.load()
    deepbach.cuda()

    # launch the script
    # use threaded=True to fix Chrome/Chromium engine hanging on requests
    # [https://stackoverflow.com/a/30670626]
    local_only = False
    if local_only:
        # accessible only locally:
        app.run(threaded=True)
    else:
        # accessible from outside:
        app.run(host='0.0.0.0', port=port, threaded=True)


def get_fermatas_tensor(metadata_tensor: torch.Tensor) -> torch.Tensor:
    """
    Extract the fermatas tensor from a metadata tensor
    """
    fermatas_index = [m.__class__ for m in metadatas].index(
        FermataMetadata().__class__)
    # fermatas are shared across all voices so we only consider the first voice
    soprano_voice_metadata = metadata_tensor[0]

    # `soprano_voice_metadata` has shape
    # `(sequence_duration, len(metadatas + 1))`  (accouting for the voice
    # index metadata)
    # Extract fermatas for all steps
    return soprano_voice_metadata[:, fermatas_index]


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def compose_from_scratch():
    """
    Return a new, generated sheet
    Usage:
        - Request: empty, generation is done in an unconstrained fashion
        - Response: a sheet, MusicXML
    """
    global deepbach
    global _sequence_length_ticks
    global _num_iterations
    global _tensor_sheet
    global _tensor_metadata

    # Use more iterations for the initial generation step
    # FIXME hardcoded 4/4 time-signature
    num_measures_generation = math.floor(_sequence_length_ticks /
                                         deepbach.dataset.subdivision)
    initial_num_iterations = math.floor(_num_iterations * num_measures_generation
                                        / 3)  # HACK hardcoded reduction

    (generated_sheet, _tensor_sheet, _tensor_metadata) = (
        deepbach.generation(num_iterations=initial_num_iterations,
                            sequence_length_ticks=_sequence_length_ticks)
    )
    return generated_sheet


@app.route('/compose', methods=['POST'])
def compose():
    try:
    global deepbach
    global _num_iterations
    global _sequence_length_ticks
    global _tensor_sheet
    global _tensor_metadata
    global bach_chorales_dataset

    NUM_MIDI_TICKS_IN_SIXTEENTH_NOTE = 120

    start_tick_selection = int(float(request.form['start_tick']) / NUM_MIDI_TICKS_IN_SIXTEENTH_NOTE)
    end_tick_selection = int(float(request.form['end_tick']) / NUM_MIDI_TICKS_IN_SIXTEENTH_NOTE)

    file_path = request.form['file_path']
    print("file_path:", file_path, flush=True)
    print("exists:", os.path.exists(file_path), flush=True)
    if os.path.exists(file_path):
        print("size:", os.path.getsize(file_path), flush=True)

    root, ext = os.path.splitext(file_path)
    assert ext.lower() == '.mxl', f"Expected .mxl, got {ext}"

    # On écrit l'XML à côté du .mxl
    # xml_file = f"{root}.xml"
    xml_file = os.path.join(tempfile.gettempdir(), "deepbach.xml")

    # Si pas de sélection : régénération complète
    if start_tick_selection == 0 and end_tick_selection == 0:
        generated_sheet = compose_from_scratch()
        # plus explicite/robuste que 'xml'
        generated_sheet.write('musicxml', fp=xml_file)
        return sheet_to_response(generated_sheet)

    # --- Extraction MXL -> XML (portable, sans unzip) ---
    # .mxl est un zip contenant au moins un .xml (et parfois un META-INF)
    with zipfile.ZipFile(file_path, 'r') as z:
        
        xml_members = [n for n in z.namelist() if n.lower().endswith('.xml')]

        if not xml_members:
            return "No .xml found inside the .mxl archive", 400

        # Heuristique: prendre le plus gros xml (souvent le fichier principal)
        main_xml = max(xml_members, key=lambda n: z.getinfo(n).file_size)

        with zipfile.ZipFile(file_path, 'r') as z:
            names = z.namelist()
            print("zip entries:", names, flush=True)

        # Extraire le contenu XML dans root.xml
        with z.open(main_xml) as src, open(xml_file, 'wb') as dst:
            dst.write(src.read())

    # Parser l'XML extrait
    music21_parsed_chorale = converter.parse(xml_file)

    _tensor_sheet, _tensor_metadata = bach_chorales_dataset.transposed_score_and_metadata_tensors(
        music21_parsed_chorale, semi_tone=0
    )

    start_voice_index = int(request.form['start_staff'])
    end_voice_index = int(request.form['end_staff']) + 1

    time_index_range_ticks = [start_tick_selection, end_tick_selection]
    region_length = end_tick_selection - start_tick_selection

    # compute batch_size_per_voice:
    if region_length <= 8:
        batch_size_per_voice = 2
    elif region_length <= 16:
        batch_size_per_voice = 4
    else:
        batch_size_per_voice = 8

    num_total_iterations = int(_num_iterations * region_length / batch_size_per_voice)

    fermatas_tensor = get_fermatas_tensor(_tensor_metadata)

    # --- Generate ---
    (output_sheet, _tensor_sheet, _tensor_metadata) = deepbach.generation(
        tensor_chorale=_tensor_sheet,
        tensor_metadata=_tensor_metadata,
        temperature=1.,
        batch_size_per_voice=batch_size_per_voice,
        num_iterations=num_total_iterations,
        sequence_length_ticks=_sequence_length_ticks,
        time_index_range_ticks=time_index_range_ticks,
        fermatas=fermatas_tensor,
        voice_index_range=[start_voice_index, end_voice_index],
        random_init=True
    )

    # Écrire l'XML de sortie à côté du .mxl
    output_sheet.write('musicxml', fp=xml_file)

    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        return tb, 500
    # Garder ton comportement actuel
    return sheet_to_response(sheet=output_sheet)
def insert_musicxml_metadata(sheet: music21.stream.Stream):
    """
    Insert various metadata into the provided XML document
    The timesignature in particular is required for proper MIDI conversion
    """
    global timesignature

    from music21.clef import TrebleClef, BassClef, Treble8vbClef
    for part, name, clef in zip(
            sheet.parts,
            ['soprano', 'alto', 'tenor', 'bass'],
            [TrebleClef(), TrebleClef(), Treble8vbClef(), BassClef()]
    ):
        # empty_part = part.template()
        part.insert(0, timesignature)
        part.insert(0, clef)
        part.id = name
        part.partName = name

    md = metadata.Metadata()
    sheet.insert(0, md)

    # required for proper musicXML formatting
    sheet.metadata.title = 'DeepBach'
    sheet.metadata.composer = 'DeepBach'
    return (sheet)

def get_fermatas_tensor(metadata_tensor: torch.Tensor) -> torch.Tensor:
    """
    Extract the fermatas tensor from a metadata tensor
    """
    fermatas_index = [m.__class__ for m in metadatas].index(
        FermataMetadata().__class__)
    # fermatas are shared across all voices so we only consider the first voice
    soprano_voice_metadata = metadata_tensor[0]

    # `soprano_voice_metadata` has shape
    # `(sequence_duration, len(metadatas + 1))`  (accouting for the voice
    # index metadata)
    # Extract fermatas for all steps
    return soprano_voice_metadata[:, fermatas_index]


def sheet_to_response(sheet):
    # convert sheet to xml
    goe = musicxml.m21ToXml.GeneralObjectExporter(sheet)
    xml_chorale_string = goe.parse()
    response = make_response((xml_chorale_string, response_headers))
    return response


@app.route('/test', methods=['POST', 'GET'])
def test_generation():
    response = make_response(('TEST', response_headers))

    if request.method == 'POST':
        print(request)

    return response


@app.route('/models', methods=['GET'])
def get_models():
    models_list = ['Deepbach']
    return jsonify(models_list)


@app.route('/current_model', methods=['POST', 'PUT'])
def current_model_update():
    return 'Model is only loaded once'


@app.route('/current_model', methods=['GET'])
def current_model_get():
    return 'DeepBach'


if __name__ == '__main__':
    file_handler = logging_handlers.RotatingFileHandler(
        'app.log', maxBytes=10000, backupCount=5)

    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.run(host="127.0.0.1", port=5000, debug=True)
init_app()
