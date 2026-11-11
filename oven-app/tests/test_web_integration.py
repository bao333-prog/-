import io
import pytest
from PIL import Image
from backend.app import create_app
from oven_food.metrics import classification_metrics
from reproduce.relocate import replace_paths

@pytest.fixture(scope='module')
def client():
    return create_app().test_client()

def test_real_inference(client):
    metadata=client.get('/api/classes').json
    names=metadata['classes']
    assert metadata['version'] == metadata['model']
    assert len(names)==14 and '香肠' in names
    stream=io.BytesIO();Image.new('RGB',(64,64),'white').save(stream,format='PNG');stream.seek(0)
    result=client.post('/api/predict',data={'image':(stream,'image.png')})
    assert result.status_code==200
    assert result.json['requires_confirmation'] is True
    assert result.json['model_version'] == metadata['version']
    assert result.json['timing']['inference_ms'] >= 0
    assert len(result.json['candidates'])==3
    assert all(0<=x['confidence']<=1 and x['name'] in names for x in result.json['candidates'])
    assert client.get('/').status_code==200
    assert client.get('/camera.js').status_code==200
    assert client.get('/camera-session.js').status_code==200

def test_invalid_upload(client):
    assert client.post('/api/predict').status_code==400
    assert client.post('/api/predict',data={'image':(io.BytesIO(b'bad'),'x.png')}).status_code==400
    assert client.post('/api/predict',data={'image':(io.BytesIO(b'x'*(10*1024*1024+1)),'x.png')}).status_code==413

def test_macro():
    result=classification_metrics([0,0,0,1],[0,0,0,0],class_names=['a','b'])
    assert result['overall_accuracy']==.75 and result['macro_accuracy']==.5

def test_relocation():
    assert replace_paths({'C:\\old\\x':['C:\\old\\z']},{'C:/old':'D:/new'})=={'D:/new/x':['D:/new/z']}
