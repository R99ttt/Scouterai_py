from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Player(Base):
    __tablename__ = 'temp'
    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer)
    player_url = Column(String(255))
    fifa_version = Column(String(10))
    fifa_update = Column(String(10))
    fifa_update_date = Column(DateTime)
    short_name = Column(String(100))
    long_name = Column(String(255))
    player_positions = Column(String(255))
    overall = Column(Integer)
    potential = Column(Integer)
    value_eur = Column(Integer)
    wage_eur = Column(Integer)
    age = Column(Integer)
    dob = Column(DateTime)
    height_cm = Column(Integer)
    weight_kg = Column(Integer)
    league_id = Column(Integer)
    league_name = Column(String(100))
    league_level = Column(Integer)
    club_team_id = Column(Integer)
    club_name = Column(String(100))
    club_position = Column(String(50))
    club_jersey_number = Column(Integer)
    club_loaned_from = Column(String(100))
    club_joined_date = Column(DateTime)
    club_contract_valid_until_year = Column(Integer)
    nationality_id = Column(Integer)
    nationality_name = Column(String(100))
    nation_team_id = Column(Integer)
    nation_position = Column(String(50))
    nation_jersey_number = Column(Integer)
    preferred_foot = Column(String(10))
    weak_foot = Column(Integer)
    skill_moves = Column(Integer)
    international_reputation = Column(Integer)
    work_rate = Column(String(50))
    body_type = Column(String(50))
    real_face = Column(Boolean)
    release_clause_eur = Column(Integer)
    player_tags = Column(String(255))
    player_traits = Column(String(255))
    pace = Column(Integer)
    shooting = Column(Integer)
    passing = Column(Integer)
    dribbling = Column(Integer)
    defending = Column(Integer)
    physic = Column(Integer)
    attacking_crossing = Column(Integer)
    attacking_finishing = Column(Integer)
    attacking_heading_accuracy = Column(Integer)
    attacking_short_passing = Column(Integer)
    attacking_volleys = Column(Integer)
    skill_dribbling = Column(Integer)
    skill_curve = Column(Integer)
    skill_fk_accuracy = Column(Integer)
    skill_long_passing = Column(Integer)
    skill_ball_control = Column(Integer)
    movement_acceleration = Column(Integer)
    movement_sprint_speed = Column(Integer)
    movement_agility = Column(Integer)
    movement_reactions = Column(Integer)
    movement_balance = Column(Integer)
    power_shot_power = Column(Integer)
    power_jumping = Column(Integer)
    power_stamina = Column(Integer)
    power_strength = Column(Integer)
    power_long_shots = Column(Integer)
    mentality_aggression = Column(Integer)
    mentality_interceptions = Column(Integer)
    mentality_positioning = Column(Integer)
    mentality_vision = Column(Integer)
    mentality_penalties = Column(Integer)
    mentality_composure = Column(Integer)
    defending_marking_awareness = Column(Integer)
    defending_standing_tackle = Column(Integer)
    defending_sliding_tackle = Column(Integer)
    goalkeeping_diving = Column(Integer)
    goalkeeping_handling = Column(Integer)
    goalkeeping_kicking = Column(Integer)
    goalkeeping_positioning = Column(Integer)
    goalkeeping_reflexes = Column(Integer)
    goalkeeping_speed = Column(Integer)
    ls = Column(String(10))
    st = Column(String(10))
    rs = Column(String(10))
    lw = Column(String(10))
    lf = Column(String(10))
    cf = Column(String(10))
    rf = Column(String(10))
    rw = Column(String(10))
    lam = Column(String(10))
    cam = Column(String(10))
    ram = Column(String(10))
    lm = Column(String(10))
    lcm = Column(String(10))
    cm = Column(String(10))
    rcm = Column(String(10))
    rm = Column(String(10))
    lwb = Column(String(10))
    ldm = Column(String(10))
    cdm = Column(String(10))
    rdm = Column(String(10))
    rwb = Column(String(10))
    lb = Column(String(10))
    lcb = Column(String(10))
    cb = Column(String(10))
    rcb = Column(String(10))
    rb = Column(String(10))
    gk = Column(String(10))
    player_face_url = Column(String(255))
