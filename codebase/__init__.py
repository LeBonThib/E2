from flask import Flask

def create_app():
        app = Flask(__name__)
        app.config['SECRET_KEY'] = 'thisisasecretkey' # cookies/session variables encryption
        
        """ Establish application routes """
        from .home import home
        from .batch_predict import batch_predict

        """ Establish application blueprints """
        app.register_blueprint(home, url_prefix='/')
        app.register_blueprint(batch_predict, url_prefix='/')

        return app