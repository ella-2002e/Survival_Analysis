import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
from lifelines import WeibullFitter, ExponentialFitter, LogNormalFitter, LogLogisticFitter, LogNormalAFTFitter
import matplotlib.pyplot as plt
from scipy.stats import lognorm


class AFTModelSelector:
    """
    A class for selecting the best AFT (Accelerated Failure Time) model among Weibull, Exponential,
    Log-Normal, and Log-Logistic models based on AIC, and generating churn rate and customer lifetime value (CLV) 
    predictions for a specified number of time periods.

    Parameters:
    - data (pd.DataFrame): The input DataFrame containing survival data.
    - primary_col(str): The column name in the DataFrame representing the primary key.
    - duration_col (str): The column name in the DataFrame representing the duration or time-to-event.
    - event_col (str): The column name in the DataFrame representing the event indicator.

    Attributes:
    - data (pd.DataFrame): The input DataFrame containing survival data.
    - primary(str): The column name in the DataFrame representing the primary key.
    - duration_col (str): The column name in the DataFrame representing the duration or time-to-event.
    - event_col (str): The column name in the DataFrame representing the event indicator.
    - aft_model (lifelines.Fitter): The selected AFT model based on AIC.
    - predictions_df (pd.DataFrame): DataFrame containing churn and CLV predictions for a specified number of time periods.
    """
    
    def __init__(self, data: pd.DataFrame , primary_col:str,  duration_col : str, event_col: str):
        self.data = data
        self.primary = primary_col
        self.duration_col = duration_col
        self.event_col = event_col
        self.aft_model = None
        self.predictions_df = None
        self.clv_prediction = None

            
            
    def select_best_model(self):
        """
        Selects the best AFT model among Weibull, Exponential, Log-Normal, and Log-Logistic models based on AIC.
        Stores the selected model in the 'aft_model' attribute.
        """
        models = {
            'Weibull': WeibullFitter(),
            'Exponential': ExponentialFitter(),
            'LogNormal': LogNormalAFTFitter(),
            'LogLogistic': LogLogisticFitter(),
        }

        aic_values = {}
        survival_functions = {}

        # Handle zero values in the duration column
        self.data[self.duration_col] = self.data[self.duration_col].replace(0, 0.0001)

        for model_name, model in models.items():
            if model_name == 'LogNormal':
                model.fit(self.data, duration_col=self.duration_col, event_col=self.event_col)
                models[model_name] = model
            else:
                model.fit(durations=self.data[self.duration_col], event_observed=self.data[self.event_col])
                models[model_name] = model

            aic = model.AIC_
            aic_values[model_name] = aic

            # Store survival functions
            if model_name in ['Weibull', 'Exponential','LogLogistic']:
                survival_functions[model_name] = model.survival_function_
            elif model_name == 'LogNormal':
                # For LogNormal which has predict_survival_function
                predictions = model.predict_survival_function(self.data)
                survival_functions[model_name] = predictions
            else:
                raise AttributeError(f"{model_name} has no attribute 'survival_function_' or 'predict_survival_function'.")

        # Find the model with the minimum AIC
        best_model = min(aic_values, key=aic_values.get)

        # Store AIC values and survival functions in self
        self.aic_values = aic_values
        self.models = models
        self.survival_functions = survival_functions

        # Store the selected model
        self.aft_model = models[best_model]




    def fit_and_predict(self, n_time_periods: int):
        """
        Fits the selected AFT model and generates churn predictions for a specified number of time periods.
        Stores the predictions in the 'predictions_df' attribute.

        Parameters:
        - n_time_periods (int): The number of time periods for which predictions should be generated.
            
        Returns:
        - str: A message indicating the model ran successfully. 
        """
        if self.aft_model is None:
            return

        # Handle zero values in the duration column
        self.data[self.duration_col] = self.data[self.duration_col].replace(0, 0.0001)

        predictions_df_list = []

        for time_period in range(1, n_time_periods + 1):
            customer_data = pd.DataFrame({
                'customer_id': self.data[self.primary],
                'pred_period': time_period
            })

            # Generate survival predictions 
            predictions = self.aft_model.predict_survival_function(self.data, times=[time_period])

            #obtaining churn predictions
            churn = round(1 - predictions, 5)
            # Convert predictions to a DataFrame
            predictions_df = pd.DataFrame(churn.T.values, columns=['churn_rate'])

            # Concatenate customer_id and time_period with predictions DataFrame
            result_df = pd.concat([customer_data, predictions_df], axis=1)

            # Append to the list
            predictions_df_list.append(result_df)

        # Concatenate all predictions into a single DataFrame
        self.predictions_df = pd.concat(predictions_df_list, ignore_index=True)

    def calculate_clv(self, MM=1300, r=0.1):
        """
        Calculate Customer Lifetime Value (CLV) for each customer in the predictions DataFrame.

        Parameters:
            - MM (float): A constant representing the monetary value.
            - r (float): The periodic interest rate for discounting.

        Returns:
            - pd.Series: Series containing CLV values for each customer.
        """
        if self.predictions_df is None:
            return
        
        #Preparing data for calculating CLV
        self.clv_prediction = self.predictions_df.pivot(index='customer_id', columns='pred_period', values='churn_rate')
        #Calculating again the Survival rates from Churn rates df
        self.clv_prediction = 1 - self.clv_prediction
        data_clv = self.clv_prediction
        #Calculating clv
        sequence = list(range(1, len(data_clv.columns) + 1))

        # Iterating over each column in data_clv
        for num in sequence:
            # Discount the values in the column based on time-value-of-money calculation
            data_clv.iloc[:, num - 1] /= (1 + r/12) ** (num - 1)

        # Calculate CLV for each row
        data_clv['CLV'] = MM * data_clv.sum(axis=1)
        self.clv_prediction['CLV'] = data_clv['CLV']

    def plot_survival_functions(self):
        """
        Plot survival functions for all models.
        """
        plt.figure(figsize=(10, 6))
        for model_name, survival_function in self.survival_functions.items():
            plt.plot(survival_function.index, survival_function.mean(1), label=model_name)

        plt.title('Survival Functions for AFT Models')
        plt.xlabel('Time')
        plt.ylabel('Survival Probability')
        plt.legend()
        plt.show()
