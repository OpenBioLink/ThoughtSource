from flask import make_response, request

CORS_ORIGIN = 'http://localhost:3000'

def cors_handling(base_function):
    def corse_response(*args, **kwargs):
      if request.method == 'OPTIONS':
        return _build_cors_preflight_response()
      return _corsify_actual_response(base_function())

    # Renaming the function name:
    corse_response.__name__ = base_function.__name__
    return corse_response

def _build_cors_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "http://localhost:3000")
    response.headers.add("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept")
    response.headers.add("Access-Control-Allow-Methods", "PUT, POST, PATCH, DELETE, GET")
    response.headers.add("Access-Control-Allow-Credentials", "true")

    return response

def _corsify_actual_response(base_response):
    response = make_response(base_response)
    response.headers.add("Access-Control-Allow-Origin", "http://localhost:3000")
    response.headers.add("Access-Control-Allow-Credentials", "true")
    return response
