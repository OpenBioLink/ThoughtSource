## ThoughtSource⚡ Annotator Backend

Flask app to support the ThoughtSource⚡ Annotator react app. Start by running 
```
flask run --host=0.0.0.0
```
when running the application on a public URL (applicable when hosting the backend for public use), or
```
flask run
```
when accessing via localhost (applicable during local development).

Prerequisites are a Python 3 installation and the dependencies listed in requirements.txt (installed via "pip install -r requirements.txt", usually done in a virtual environment, see https://docs.python.org/3/library/venv.html).

A .env file is needed for configuration, containing values for FRONTEND_URL and BACKEND_URL (URL of the react- and flask applications, respectively).

An example configuration "example.env" can be found in the project root. In a localhost development environment, the contents of example.env can be used as is.


## Functionality

Provides sentence-to-sentence similarities to help the frontend visualise similar sentences between different CoT outputs.
