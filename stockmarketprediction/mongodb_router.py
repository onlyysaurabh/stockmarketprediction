from django.conf import settings

class MongoDBRouter:
    """
    A router to control all database operations on models.
    """
    def db_for_read(self, model, **hints):
        """
        Attempts to read auth models go to default db.
        """
        if model._meta.app_label == 'auth':
            return 'default'
        return None

    def db_for_write(self, model, **hints):
        """
        Attempts to write auth models go to default db.
        """
        if model._meta.app_label == 'auth':
            return 'default'
        return None

    def allow_relation(self, obj1, obj2, **hints):
        """
        Allow relations if both objects are in the same database.
        """
        return True

    def allow_migrate(self, db, app_label, model_name=None, **hints):
        """
        Make sure the auth app only appears in the 'default' database.
        """
        if app_label == 'auth':
            return db == 'default'
        return None