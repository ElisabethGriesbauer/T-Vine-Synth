from pandas import DataFrame, concat
import random

from utils.logging import LOGGER

# from generative_models.generative_model import GenerativeModel
# from sdv.single_table import CTGANSynthesizer
# from sdv.metadata import SingleTableMetadata


# class CTGAN(GenerativeModel):
#     """A conditional generative adversarial network for tabular data"""
#     def __init__(self, metadata,
#                  enforce_rounding=False,
#                  epochs=900, # tuned by hand for real data I d20
#                  verbose=True,
#                  batch_size=500,
#                  dis_dim=(256, 256),
#                  discriminator_decay=1e-6,
#                  discriminator_lr=2e-4,
#                  discriminator_steps=1,
#                  embedding_dim=128, 
#                  generator_decay=1e-6,
#                  # l2scale=1e-6,
#                  gen_dim=(256, 256),
#                  generator_lr=2e-4,
#                  log_frequency = True,
#                  pac = 10,
#                  multiprocess=False):

#         self.metadata = metadata
#         self.enforce_rounding = enforce_rounding
#         self.epochs = epochs
#         self.verbose = verbose
#         self.batch_size = batch_size
#         self.dis_dim = dis_dim
#         self.discriminator_decay = discriminator_decay
#         self.discriminator_lr = discriminator_lr
#         self.discriminator_steps = discriminator_steps
#         self.embedding_dim = embedding_dim
#         self.generator_decay = generator_decay
#         self.gen_dim = gen_dim
#         self.generator_lr = generator_lr
#         self.log_frequency = log_frequency
#         self.pac = pac

#         self.datatype = DataFrame

#         self.multiprocess = bool(multiprocess)

#         self.infer_ranges = True
#         self.trained = False

#         self.__name__ = 'CTGAN'

#     def fit(self, data, *args):
#         """Train a generative adversarial network on tabular data.
#         Input data is assumed to be of shape (n_samples, n_features)
#         See https://github.com/DAI-Lab/SDGym for details"""
#         assert isinstance(data, self.datatype), f'{self.__class__.__name__} expects {self.datatype} as input data but got {type(data)}'

#         metadata = SingleTableMetadata()
#         metadata.detect_from_dataframe(data=data)

#         self.synthesiser = CTGANSynthesizer(metadata=metadata,
#                  enforce_rounding=self.enforce_rounding,
#                  epochs=self.epochs,
#                  verbose=self.verbose,
#                  batch_size=self.batch_size,
#                  discriminator_dim=self.dis_dim,
#                  discriminator_decay=self.discriminator_decay,
#                  discriminator_lr=self.discriminator_lr,
#                  discriminator_steps=self.discriminator_steps,
#                  embedding_dim=self.embedding_dim, 
#                  generator_decay=self.generator_decay,
#                  generator_dim=self.gen_dim,
#                  generator_lr=self.generator_lr,
#                  log_frequency = self.log_frequency,
#                  pac = self.pac)
        
#         # if len(args) > 0:
#         #     # Merge the additional data frames using pandas.concat or another appropriate method
#         #     data = concat([data] + list(args), axis=0, ignore_index=True)
        

#         LOGGER.debug(f'Start fitting {self.__class__.__name__} to data of shape {data.shape}...')
#         self.synthesiser.fit(data)

#         LOGGER.debug(f'Finished fitting')
#         self.trained = True

#         return self

#     def generate_samples(self, nsamples):
#         """Generate random samples from the fitted Gaussian distribution"""
#         assert self.trained, "Model must first be fitted to some data."

#         LOGGER.debug(f'Generate synthetic dataset of size {nsamples}')
#         randint = random.randint(1, 100000000000000000000)

#         ## use first alternative for MIA:
#         synthetic_data = self.synthesiser.sample(num_rows=nsamples, output_file_path=f'./tmp_samples/temp{randint}')
#         # synthetic_data = self.synthesiser.sample(num_rows=nsamples, output_file_path=None)

#         return synthetic_data
    
#     def set_params(self, **params):
#         for param, value in params.items():
#             if hasattr(self, param):
#                 setattr(self, param, value)
#             else:
#                 raise ValueError(f"Invalid parameter: {param}")
            
    
#     def transform(self, X):
#         # You might need to adjust this logic based on your specific generative model
#         randint = random.randint(1, 100000000000000000000)
#         return self.synthesiser.sample(num_rows=len(X), output_file_path=f'./tmp_samples/temp{randint}')


#------------------------------


from pandas import DataFrame
import os
os.chdir('./synthetic_data_release-master')
from generative_models.tvae import TVAE

import plotly.express as px

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from utils.datagen import load_local_data_as_df

rawPop, metadata = load_local_data_as_df('./data/real_data_I_d20')

# Assuming the last column is the target variable
X = rawPop.iloc[:, :-1]
y = rawPop.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the CTGAN model as a step in the pipeline
tvae = TVAE(metadata)
classifier = RandomForestClassifier()

# Create a pipeline with CTGAN and a classifier (you can modify this based on your needs)
pipeline = Pipeline([
    ('tvae', tvae),
    ('classifier', classifier)
])

# Define the hyperparameter grid for CTGAN
param_dist = {
    'tvae__epochs': [i for i in range(100, 1501, 100)],
    'tvae__batch_size': [i for i in range(100, 501, 100)],
    'tvae__embedding_dim':  [i for i in range(2,11)]
    # Add more parameters as needed
}

# # Perform grid search
# grid_search = GridSearchCV(
#     pipeline,
#     param_grid=param_dist,
#     scoring='roc_auc',  # Use an appropriate scoring metric
#     cv=5,  # Adjust the number of cross-validation folds as needed
#     verbose=1,
#     n_jobs=-1  # Use all available CPU cores
# )

# # Fit the grid search to your data
# random.seed(123)
# grid_search.fit(X_train, y_train)

# results = DataFrame(grid_search.cv_results_)
# results[[ 'params', 'mean_test_score', 'std_test_score', 'rank_test_score']]

# # Print the best hyperparameters
# print("Best Hyperparameters:", grid_search.best_params_)

# # Evaluate the model on the test set
# auc = grid_search.score(X_test, y_test)
# print("Test Accuracy:", auc)





# Perform random search
random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=10,  # Adjust the number of iterations as needed
    scoring='roc_auc',  # Use an appropriate scoring metric
    cv=5,  # Adjust the number of cross-validation folds as needed
    verbose=1,
    n_jobs=-1,  # Use all available CPU cores
    random_state=320
)

# Fit the random search to your data
random_search.fit(X_train, y_train)

results = DataFrame(random_search.cv_results_)[[ 'params',
       'mean_test_score', 'std_test_score', 'rank_test_score']]

# Print the best hyperparameters
print("Best Hyperparameters:", random_search.best_params_)

# Evaluate the model on the test set
auc = random_search.score(X_test, y_test)
print("Test Accuracy:", auc)




# ---------------------------------------------
## checking by hand:
from pandas import DataFrame
import os
os.chdir('./synthetic_data_release-master')

from generative_models.tvae import TVAE
import plotly.express as px
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV

from utils.datagen import load_local_data_as_df

# generate loss plot for paper with the parameter sets up I settled on:

rawPop, metadata = load_local_data_as_df('./data/real_data_I_d20')


tvae = TVAE(metadata,
   epochs=1500,
   batch_size = 400,
   embedding_dim=2)

real_train, real_test = train_test_split(rawPop, test_size=0.2, random_state=42)

random.seed(456)
tvae.fit(data=real_train)

loss_values = tvae.synthesiser._model.loss_values

# loss_values_reformatted = pd.melt(
#    loss_values,
#    id_vars=['Epoch'],
#    var_name='Loss Type'
# )

# fig = px.line(loss_values_reformatted, x="Epoch", y="value", color="Loss Type", title='Epoch vs. Loss')

fig = px.line(loss_values, x="Epoch", y="Loss", title='Epoch vs. Loss')
fig.show()


synthData = tvae.generate_samples(1000)
synthData['Y'].describe()

X = synthData.iloc[:, :-1]
y = synthData.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_hat = clf.predict(real_test.iloc[:, :-1])
sum(real_test.iloc[:, -1] == y_hat)/len(y_hat)










################## for support2 data:

from pandas import DataFrame, concat
import random

from utils.logging import LOGGER

from pandas import DataFrame
import os
os.chdir('./synthetic_data_release-master')
from generative_models.tvae import TVAE

import plotly.express as px

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from utils.datagen import load_local_data_as_df

rawPop, metadata = load_local_data_as_df('./data/real_support2')

# Assuming the last column is the target variable
X = rawPop.iloc[:, :-1]
y = rawPop.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the CTGAN model as a step in the pipeline
tvae = TVAE(metadata)
classifier = RandomForestClassifier()

# Create a pipeline with CTGAN and a classifier (you can modify this based on your needs)
pipeline = Pipeline([
    ('tvae', tvae),
    ('classifier', classifier)
])

# Define the hyperparameter grid for CTGAN
param_dist = {
    'tvae__epochs': [i for i in range(100, 1501, 100)],
    'tvae__batch_size': [i for i in range(100, 501, 100)],
    'tvae__embedding_dim':  [i for i in range(2,11)]
    # Add more parameters as needed
}


# Perform random search
random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=10,  # Adjust the number of iterations as needed
    scoring='roc_auc',  # Use an appropriate scoring metric
    cv=5,  # Adjust the number of cross-validation folds as needed
    verbose=1,
    n_jobs=-1,  # Use all available CPU cores
    random_state=420
)

# Fit the random search to your data
random_search.fit(X_train, y_train)

results = DataFrame(random_search.cv_results_)[[ 'params',
       'mean_test_score', 'std_test_score', 'rank_test_score']]

# Print the best hyperparameters
print("Best Hyperparameters:", random_search.best_params_)

# Evaluate the model on the test set
random.seed(485)
auc = random_search.score(X_test, y_test)
print("Test Accuracy:", auc)




# ---------------------------------------------
## checking by hand:

tvae = TVAE(metadata,
   epochs=500,
   batch_size = 300,
   embedding_dim=7)

real_train, real_test = train_test_split(rawPop, test_size=0.2, random_state=42)

random.seed(456)
tvae.fit(data=real_train)

loss_values = tvae.synthesiser._model.loss_values

fig = px.line(loss_values, x="Epoch", y="Loss", title='Epoch vs. Loss')
fig.show()


synthData = tvae.generate_samples(1000)
synthData['Y'].describe()

X = synthData.iloc[:, :-1]
y = synthData.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_hat = clf.predict(real_test.iloc[:, :-1])
sum(real_test.iloc[:, -1] == y_hat)/len(y_hat)









################## for support2 SMALL data:

from pandas import DataFrame, concat
import random

from utils.logging import LOGGER

from pandas import DataFrame
import os
os.chdir('./synthetic_data_release-master')
from generative_models.tvae import TVAE

import plotly.express as px

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from utils.datagen import load_local_data_as_df

rawPop, metadata = load_local_data_as_df('./data/real_support2_small')

# Assuming the last column is the target variable
X = rawPop.iloc[:, :-1]
y = rawPop.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the CTGAN model as a step in the pipeline
tvae = TVAE(metadata)
classifier = RandomForestClassifier()

# Create a pipeline with CTGAN and a classifier (you can modify this based on your needs)
pipeline = Pipeline([
    ('tvae', tvae),
    ('classifier', classifier)
])

# Define the hyperparameter grid for CTGAN
param_dist = {
    'tvae__epochs': [i for i in range(100, 1501, 100)],
    'tvae__batch_size': [i for i in range(100, 501, 100)],
    'tvae__embedding_dim':  [i for i in range(2,11)]
    # Add more parameters as needed
}


# Perform random search
random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=15,  # Adjust the number of iterations as needed
    scoring='roc_auc',  # Use an appropriate scoring metric
    cv=5,  # Adjust the number of cross-validation folds as needed
    verbose=1,
    n_jobs=-1,  # Use all available CPU cores
    random_state=420
)

# Fit the random search to your data
random_search.fit(X_train, y_train)

results = DataFrame(random_search.cv_results_)[[ 'params',
       'mean_test_score', 'std_test_score', 'rank_test_score']]

# Print the best hyperparameters
print("Best Hyperparameters:", random_search.best_params_)

# Evaluate the model on the test set
random.seed(485)
auc = random_search.score(X_test, y_test)
print("Test Accuracy:", auc)




# ---------------------------------------------
## checking by hand:

tvae = TVAE(metadata,
   epochs=800,
   batch_size = 100,
   embedding_dim=4)

real_train, real_test = train_test_split(rawPop, test_size=0.2, random_state=42)

random.seed(456)
tvae.fit(data=real_train)

loss_values = tvae.synthesiser._model.loss_values

fig = px.line(loss_values, x="Epoch", y="Loss", title='Epoch vs. Loss')
fig.show()


synthData = tvae.generate_samples(1000)
synthData['Y'].describe()

X = synthData.iloc[:, :-1]
y = synthData.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_hat = clf.predict(real_test.iloc[:, :-1])
sum(real_test.iloc[:, -1] == y_hat)/len(y_hat)
