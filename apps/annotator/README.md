## ThoughtSource⚡ Annotator

A web application to visualise and evaluate generated chains of thought. It works standalone, and is optionally enhanced with sentence similarity visualisations when running the backend flask server.

The annotator can be started by running
```
npm start
```
from the command line in the annotator folder (/apps/annotator).

Prerequisites are a nodeJS installation, and installing all project dependencies (by running "npm install").

A .env file is needed for configuration, containing values for REACT_APP_FRONTEND_URL and REACT_APP_BACKEND_URL (URL of the react- and flask applications, respectively).

An example configuration "example.env" can be found in the project root. In a localhost development environment, the contents of example.env can be used as is.
