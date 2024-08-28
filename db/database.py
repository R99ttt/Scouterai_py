from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

def get_engine():
    try:
        print("Connecting to the database...")
        engine = create_engine(
            'mysql+pymysql://root:password@localhost:3306/player_data_db',
            echo=True,
            connect_args={"connect_timeout": 10}
        )
        print("Database connection successful.")
        return engine
    except Exception as e:
        print("Failed to connect to the database.")
        print(f"Error: {e}")
        raise e

def get_session(engine):
    Session = sessionmaker(bind=engine)
    return Session()
