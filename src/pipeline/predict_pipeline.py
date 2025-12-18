import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.base import clone


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
             model_path='artifacts/model.pkl'
             preprocessor_path ='artifacts/preprocessor.pkl'
             model=load_object(file_path=model_path)
             preprocessor = load_object(file_path=preprocessor_path)
             data_scaled=preprocessor.transform(features)
             preds=model.predict(data_scaled)
             return preds
        except Exception as e:
            raise CustomException(e,sys)




class CustomData:
    def __init__(self, CityTier, WarehouseToHome, HourSpendOnApp,
                 NumberOfDeviceRegistered, SatisfactionScore, NumberOfAddress,
                 Complain, OrderAmountHikeFromlastYear, CouponUsed, OrderCount,
                   DaySinceLastOrder, CashbackAmount, Gender, PreferredLoginDevice,
                     PreferredPaymentMode, PreferedOrderCat,Tenure,MaritalStatus, high_price):
        self.CityTier = CityTier
        self.WarehouseToHome = WarehouseToHome
        self.HourSpendOnApp = HourSpendOnApp
        self.NumberOfDeviceRegistered = NumberOfDeviceRegistered
        self.SatisfactionScore = SatisfactionScore
        self.NumberOfAddress = NumberOfAddress
        self.Complain = Complain
        self.OrderAmountHikeFromlastYear = OrderAmountHikeFromlastYear
        self.CouponUsed = CouponUsed
        self.OrderCount = OrderCount
        self.DaySinceLastOrder = DaySinceLastOrder
        self.CashbackAmount = CashbackAmount
        self.Gender = Gender
        self.PreferredLoginDevice = PreferredLoginDevice
        self.PreferredPaymentMode = PreferredPaymentMode
        self.PreferedOrderCat = PreferedOrderCat
        self.MaritalStatus = MaritalStatus
        self.Tenure = Tenure
        self.high_price = high_price




    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "CityTier": [self.CityTier],
                "WarehouseToHome": [self.WarehouseToHome],
                "HourSpendOnApp": [self.HourSpendOnApp],
                "NumberOfDeviceRegistered": [self.NumberOfDeviceRegistered],
                "SatisfactionScore": [self.SatisfactionScore],
                "NumberOfAddress": [self.NumberOfAddress],
                "Complain": [self.Complain],
                "OrderAmountHikeFromlastYear": [self.OrderAmountHikeFromlastYear],
                "CouponUsed": [self.CouponUsed],
                "OrderCount": [self.OrderCount],
                "DaySinceLastOrder": [self.DaySinceLastOrder],
                "CashbackAmount": [self.CashbackAmount],
                "Gender": [self.Gender],
                "PreferredLoginDevice": [self.PreferredLoginDevice],
                "PreferredPaymentMode": [self.PreferredPaymentMode],
                "PreferedOrderCat": [self.PreferedOrderCat],
                "MaritalStatus": [self.MaritalStatus],
                "Tenure": [self.Tenure],
                "high_price": [self.high_price]

            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name, model in models.items():
            params = param[model_name]

            # Hyperparameter tuning
            gs = GridSearchCV(model, para, cv=3, n_jobs=-1, verbose=0)
            gs.fit(X_train, y_train)

            # Set best params
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Classification score (accuracy)
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)

            # Save only test accuracy
            report[list(models.keys())[i]] = test_acc

        return report

    except Exception as e:
        raise CustomException(e, sys)
          