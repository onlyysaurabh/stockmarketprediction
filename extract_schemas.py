#!/usr/bin/env python3
import os
import sys
import json
import sqlite3
from pathlib import Path

# Add the project directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stockmarketprediction.settings')
import django
django.setup()

from django.conf import settings
from django.db import connections
import pymongo

# Setup output structure
output = {
    "sqlite_schema": [],
    "mongodb_collections": {}
}

def extract_sqlite_schema():
    """Extract schema from SQLite database"""
    try:
        print(f"Connecting to SQLite DB: {settings.DATABASES['default']['NAME']}")
        if not os.path.exists(settings.DATABASES['default']['NAME']):
            print(f"SQLite database file not found: {settings.DATABASES['default']['NAME']}")
            return False
            
        conn = sqlite3.connect(settings.DATABASES['default']['NAME'])
        cursor = conn.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Found {len(tables)} tables in SQLite database")
        
        for table in tables:
            table_name = table[0]
            if table_name.startswith('sqlite_'):
                continue  # Skip SQLite internal tables
            
            print(f"Processing table: {table_name}")
            # Get table schema
            cursor.execute(f"PRAGMA table_info('{table_name}');")
            columns = cursor.fetchall()
            
            # Get foreign keys
            cursor.execute(f"PRAGMA foreign_key_list('{table_name}');")
            foreign_keys = cursor.fetchall()
            
            # Get indices
            cursor.execute(f"PRAGMA index_list('{table_name}');")
            indices = cursor.fetchall()
            
            table_schema = {
                "table_name": table_name,
                "columns": [],
                "foreign_keys": [],
                "indices": []
            }
            
            for column in columns:
                col_id, name, data_type, not_null, default_value, is_pk = column
                table_schema["columns"].append({
                    "name": name,
                    "type": data_type,
                    "not_null": bool(not_null),
                    "default": default_value,
                    "is_primary_key": bool(is_pk)
                })
            
            for fk in foreign_keys:
                id, seq, table_ref, from_col, to_col, on_update, on_delete, match = fk
                table_schema["foreign_keys"].append({
                    "from_column": from_col,
                    "to_table": table_ref,
                    "to_column": to_col,
                    "on_update": on_update,
                    "on_delete": on_delete
                })
            
            for idx in indices:
                idx_name, unique, origin, partial = idx[:4]
                cursor.execute(f"PRAGMA index_info('{idx_name}')")
                idx_columns = cursor.fetchall()
                columns = [col[2] for col in idx_columns]
                table_schema["indices"].append({
                    "name": idx_name,
                    "unique": bool(unique),
                    "columns": columns
                })
            
            output["sqlite_schema"].append(table_schema)
        
        conn.close()
        return True
    except Exception as e:
        print(f"Error extracting SQLite schema: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def extract_mongodb_schema():
    """Extract schema definition from MongoDB collections without returning sample data"""
    try:
        # Connect to MongoDB
        print(f"Connecting to MongoDB: {settings.MONGO_URI}")
        client = pymongo.MongoClient(settings.MONGO_URI)
        
        # Test connection
        client.admin.command('ping')
        print("MongoDB connection successful")
        
        db = client[settings.MONGO_DB_NAME]
        print(f"Using MongoDB database: {settings.MONGO_DB_NAME}")
        
        # Get all collection names
        collection_names = db.list_collection_names()
        print(f"Found {len(collection_names)} collections in MongoDB")
        
        if len(collection_names) == 0:
            print("No collections found in MongoDB")
            return False
        
        for collection_name in collection_names:
            print(f"Processing collection: {collection_name}")
            # Analyze collection schema without returning sample documents
            collection = db[collection_name]
            
            # Get fields and their types by sampling a few documents
            fields = {}
            pipeline = [{"$sample": {"size": 5}}]
            
            try:
                sample_docs = list(collection.aggregate(pipeline))
                print(f"  Sampled {len(sample_docs)} documents from {collection_name}")
                
                # Combine field information from all sample docs
                for doc in sample_docs:
                    for field_name, value in doc.items():
                        # Skip _id field type as it's always ObjectId
                        if field_name == '_id':
                            fields[field_name] = 'ObjectId'
                            continue
                        # Get the type of the field
                        field_type = type(value).__name__
                        # If we've seen this field before, only update if the type differs
                        if field_name in fields and fields[field_name] != field_type:
                            fields[field_name] = 'mixed'
                        else:
                            fields[field_name] = field_type
            except Exception as e:
                print(f"  Error sampling documents from {collection_name}: {str(e)}")
            
            # Get collection stats and index information
            try:
                stats = db.command('collStats', collection_name)
                print(f"  Got stats for {collection_name}")
            except Exception as e:
                print(f"  Error getting stats for {collection_name}: {str(e)}")
                stats = {}
            
            try:
                indexes = list(collection.list_indexes())
                print(f"  Found {len(indexes)} indexes for {collection_name}")
            except Exception as e:
                print(f"  Error getting indexes for {collection_name}: {str(e)}")
                indexes = []
            
            # Extract index information in a simplified format
            index_info = []
            for index in indexes:
                index_info.append({
                    "name": index.get('name'),
                    "key_fields": list(index.get('key', {}).items()),
                    "unique": index.get('unique', False)
                })
            
            # Final collection schema information
            output["mongodb_collections"][collection_name] = {
                "fields": fields,
                "count": stats.get('count', 0),
                "size": stats.get('size', 0),
                "avg_obj_size": stats.get('avgObjSize', 0),
                "indexes": index_info
            }
        
        return True
    except Exception as e:
        print(f"Error extracting MongoDB schema: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def extract_django_models():
    """Extract Django model schema information"""
    try:
        from django.apps import apps
        print("Extracting Django model information")
        
        output["django_models"] = []
        
        for app_config in apps.get_app_configs():
            app_name = app_config.label
            print(f"Processing app: {app_name}")
            
            # Skip contrib apps 
            if app_name in ['admin', 'auth', 'contenttypes', 'sessions', 'messages']:
                continue
                
            for model in app_config.get_models():
                model_name = model.__name__
                print(f"  Processing model: {model_name}")
                
                model_info = {
                    "app": app_name,
                    "name": model_name,
                    "fields": [],
                    "meta": {
                        "db_table": model._meta.db_table,
                        "verbose_name": str(model._meta.verbose_name),
                        "verbose_name_plural": str(model._meta.verbose_name_plural),
                        "ordering": model._meta.ordering or [],
                    }
                }
                
                for field in model._meta.fields:
                    field_info = {
                        "name": field.name,
                        "type": field.__class__.__name__,
                        "primary_key": field.primary_key,
                        "unique": field.unique,
                        "blank": field.blank,
                        "null": field.null,
                    }
                    
                    if hasattr(field, 'remote_field') and field.remote_field:
                        field_info["related_model"] = field.remote_field.model.__name__
                    
                    model_info["fields"].append(field_info)
                
                output["django_models"].append(model_info)
        
        return True
    except Exception as e:
        print(f"Error extracting Django models: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Extracting database schemas...")
    output_file = os.path.join(current_dir, "database_schema.json")
    print(f"Will save output to: {output_file}")
    
    # Extract SQLite schema
    sqlite_success = extract_sqlite_schema()
    print(f"SQLite schema extraction {'successful' if sqlite_success else 'failed'}")
    
    # Extract MongoDB schema
    mongo_success = extract_mongodb_schema()
    print(f"MongoDB schema extraction {'successful' if mongo_success else 'failed'}")
    
    # Extract Django models
    django_success = extract_django_models()
    print(f"Django models extraction {'successful' if django_success else 'failed'}")
    
    # Write to file
    try:
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"Schema extraction complete. Results saved to {output_file}")
    except Exception as e:
        print(f"Error writing to output file: {str(e)}")
        import traceback
        traceback.print_exc()
