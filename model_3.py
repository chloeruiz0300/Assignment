import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from tqdm.notebook import tqdm
from biogeme.expressions import Beta, Variable, log, exp
from biogeme import models
from biogeme.results_processing import EstimationResults
from biogeme.results_processing import get_pandas_estimated_parameters, html_output

df = pd.read_csv('lpmc19.dat', sep='\t')
df['age'].dropna()

database = db.Database('netherlands', df)
trip_id = Variable('trip_id')
travel_mode = Variable('travel_mode')
purpose = Variable('purpose')
distance = Variable('distance')
dur_walking = Variable('dur_walking')
dur_cycling = Variable('dur_cycling')
dur_pt_access = Variable('dur_pt_access')
dur_pt_rail  = Variable('dur_pt_rail')
dur_pt_bus  = Variable('dur_pt_bus')
dur_pt_int = Variable('dur_pt_int')
dur_driving = Variable('dur_driving')
cost_transit = Variable('cost_transit')
cost_driving_fuel = Variable('cost_driving_fuel')
cost_driving_ccharge = Variable('cost_driving_ccharge')
driving_traffic_percent = Variable('driving_traffic_percent')
age = Variable('age')

dur_pt_total = dur_pt_access + dur_pt_rail + dur_pt_bus + dur_pt_int
cost_car = cost_driving_fuel + cost_driving_ccharge

POUNDS_TO_EUROS = 1.16
cost_car_euro = cost_car * POUNDS_TO_EUROS
cost_pt_euro = cost_transit * POUNDS_TO_EUROS

asc_car = Beta('asc_car', 0, None, None, 0)
asc_walk = Beta('asc_walk', 0, None, None, 0)
asc_bike = Beta('asc_bike', 0, None, None, 0)
asc_pt = Beta('asc_pt', 0, None, None, 0)
beta_cost = Beta('beta_cost_car', 0, None, None, 0)
beta_age = Beta('beta_age', 0, None, None, 0)
beta_time_car = Beta('beta_time_car', 0, None, None, 0)
beta_time_walk = Beta('beta_time_walk', 0, None, None, 0)
beta_time_bike = Beta('beta_time_bike', 0, None, None, 0)
beta_time_pt = Beta('beta_time_pt', 0, None, None, 0)
beta_driving_traffic_percent = Beta('beta_driving_traffic_percent', 0, None, None, 0)

v_car = asc_car + beta_age * age + beta_cost * cost_car_euro + beta_time_car * log(dur_driving) + beta_driving_traffic_percent * driving_traffic_percent
v_walk = 0 + beta_time_walk * log(dur_walking)
v_bike = asc_bike + beta_age * age + beta_time_bike * log(dur_cycling)
v_pt = asc_pt + beta_age * age + beta_cost * cost_pt_euro + beta_time_pt * log(dur_pt_total)

V = {1: v_walk, 2: v_bike, 3: v_pt, 4: v_car}
logprob = models.loglogit(V, None, travel_mode)
