#curl -u user:password -i http://localhost:5000/todo/api/v1.0/tasks
#curl -u user:password -i http://localhost:5000/todo/api/v1.0/tasks
#curl -u user:password -i -H "Content-Type: application/json" -X POST -d '{"title":"Read a book"}' http://localhost:5000/todo/api/v1.0/tasks/1

#curl -u user:password -X POST http://localhost:5000/predictions/1 -T data/example/1.jpg
# 1 is person image

from flask import Flask, jsonify, abort, request, make_response, url_for
from flask_httpauth import HTTPBasicAuth
import logging
import os

logging.basicConfig(level=logging.INFO, filename="logs/flask_logs.log",filemode="w")

UPLOAD_FOLDER = 'data/from_flask'
app = Flask(__name__, static_url_path = "")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
auth = HTTPBasicAuth()

@app.route('/predictions/<int:task_id>', methods = ['POST'])
@auth.login_required
def launch_task(task_id):
    
    logging.info(f"launch task with id:{task_id}")
    if request.method == 'POST':
        logging.info(f'request.files is {request.files.keys()}')
        
        if 'file' not in request.files:
            return 'there is no file in form!'
        file = request.files['file']
        #return "ok"
        # from matplotlib import pyplot as plt
        # import numpy as np
        # logging.info(f'file var type is {type(file1)}')
        # im = np.frombuffer(file1)
        # plt.imshow(im)

        path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(path)
    #     return path

    # image =
    # if not request.json or not 'title' in request.json:
    #     abort(400)

    # task = {
    # #     'id': tasks[-1]['id'] + 1,
    # #     'title': request.json['title'],
    # #     'description': request.json.get('description', ""),
    # #     'done': False
    # # }
    # tasks.append(task)
    
    return jsonify( { 'status': 0 } ), 201


@auth.get_password
def get_password(username):
    if username == 'user':
        return 'password'
    return None

@auth.error_handler
def unauthorized():
    return make_response(jsonify( { 'error': 'Unauthorized access' } ), 403)
    # return 403 instead of 401 to prevent browsers from displaying the default auth dialog
    
@app.errorhandler(400)
def bad_request(error):
    return make_response(jsonify( { 'error': 'Bad request' } ), 400)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify( { 'error': 'Not found' } ), 404)

tasks = [
    {
        'id': 1,
        'title': u'Load person image',
        'description': u'Load and process data of person image. Required data: image, user_id, image_id', 
        'done': False
    },
    {
        'id': 2,
        'title': u'Learn Python',
        'description': u'Need to find a good Python tutorial on the web', 
        'done': False
    }
]

def make_public_task(task):
    new_task = {}
    for field in task:
        if field == 'id':
            new_task['uri'] = url_for('get_task', task_id = task['id'], _external = True)
        else:
            new_task[field] = task[field]
    return new_task
    
@app.route('/todo/api/v1.0/tasks', methods = ['GET'])
@auth.login_required
def get_tasks():
    return jsonify( { 'tasks': list(map(make_public_task, tasks)) } )


@app.route('/todo/api/v1.0/tasks/<int:task_id>', methods = ['GET'])
@auth.login_required
def get_task(task_id):
    task = filter(lambda t: t['id'] == task_id, tasks)
    if len(task) == 0:
        abort(404)
    return jsonify( { 'task': make_public_task(task[0]) } )


# @app.route('/todo/api/v1.0/tasks', methods = ['POST'])
# @auth.login_required
# def create_task():
#     if not request.json or not 'title' in request.json:
#         abort(400)
#     task = {
#         'id': tasks[-1]['id'] + 1,
#         'title': request.json['title'],
#         'description': request.json.get('description', ""),
#         'done': False
#     }
#     tasks.append(task)
#     return jsonify( { 'task': make_public_task(task) } ), 201





# @app.route('/todo/api/v1.0/tasks/<int:task_id>', methods = ['PUT'])
# @auth.login_required
# def update_task(task_id):
#     task = filter(lambda t: t['id'] == task_id, tasks)
#     if len(task) == 0:
#         abort(404)
#     if not request.json:
#         abort(400)
#     if 'title' in request.json and type(request.json['title']) != unicode:
#         abort(400)
#     if 'description' in request.json and type(request.json['description']) is not unicode:
#         abort(400)
#     if 'done' in request.json and type(request.json['done']) is not bool:
#         abort(400)
#     task[0]['title'] = request.json.get('title', task[0]['title'])
#     task[0]['description'] = request.json.get('description', task[0]['description'])
#     task[0]['done'] = request.json.get('done', task[0]['done'])
#     return jsonify( { 'task': make_public_task(task[0]) } )
    
# @app.route('/todo/api/v1.0/tasks/<int:task_id>', methods = ['DELETE'])
# @auth.login_required
# def delete_task(task_id):
#     task = filter(lambda t: t['id'] == task_id, tasks)
#     if len(task) == 0:
#         abort(404)
#     tasks.remove(task[0])
#     return jsonify( { 'result': True } )
    
if __name__ == '__main__':
    app.run(debug = True)