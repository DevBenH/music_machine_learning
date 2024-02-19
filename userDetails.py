from flask.ext.sqlalchemy improt SQLAlchemy 

db = SQLAlchemy()

class User(db.Model):
	__tablename__ = 'User'

	username = db.Column(db.String, primary_key = True)
	password = db.Column(db.String)
	authenticated = db.Column(db.Boolean, default = False)


	def get_username(self):
		return self.username

	def is_authenticated(self):
		return self.authenticated

	def is_activte(self):
		return True


	@login_manager.user_loader
	def user_loader(id):
		return User.query.get(id)

		