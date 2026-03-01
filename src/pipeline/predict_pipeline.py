import sys
import logging
import pandas as pd

from src.exception import CustomException
from src.utils import load_object

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.base import clone


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class PredictPipeline:
    def __init__(self):
        logger.info("PredictPipeline initialized")

    def predict(self, features):
        try:
            logger.info("Loading model and preprocessor for prediction")

            model = load_object(file_path="artifacts/model.pkl")
            preprocessor = load_object(file_path="artifacts/preprocessor.pkl")

            logger.info("Transforming input features")
            data_scaled = preprocessor.transform(features)

            logger.info("Generating class prediction")
            preds = model.predict(data_scaled)

            logger.info(f"Prediction completed: {preds}")
            return preds

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise CustomException(e, sys)

    def predict_proba(self, features):
        try:
            logger.info("Loading model and preprocessor for probability prediction")

            model = load_object(file_path="artifacts/model.pkl")
            preprocessor = load_object(file_path="artifacts/preprocessor.pkl")

            logger.info("Transforming input features")
            data_scaled = preprocessor.transform(features)

            logger.info("Generating probability prediction")
            proba = model.predict_proba(data_scaled)

            logger.info(f"Prediction probabilities: {proba}")
            return proba

        except Exception as e:
            logger.error(f"Error during probability prediction: {e}")
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        CityTier,
        WarehouseToHome,
        HourSpendOnApp,
        NumberOfDeviceRegistered,
        SatisfactionScore,
        NumberOfAddress,
        Complain,
        OrderAmountHikeFromlastYear,
        CouponUsed,
        OrderCount,
        DaySinceLastOrder,
        CashbackAmount,
        Gender,
        PreferredLoginDevice,
        PreferredPaymentMode,
        PreferedOrderCat,
        Tenure,
        MaritalStatus,
        high_price,
    ):
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
        self.Tenure = Tenure
        self.Gender = Gender
        self.PreferredLoginDevice = PreferredLoginDevice
        self.PreferredPaymentMode = PreferredPaymentMode
        self.PreferedOrderCat = PreferedOrderCat
        self.MaritalStatus = MaritalStatus
        self.high_price = high_price

        logger.info("CustomData object created")

    def get_data_as_data_frame(self):
        try:
            logger.info("Converting input data to DataFrame")

            data_dict = {
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
                "Tenure": [self.Tenure],
                "Gender": [self.Gender],
                "PreferredLoginDevice": [self.PreferredLoginDevice],
                "PreferredPaymentMode": [self.PreferredPaymentMode],
                "PreferedOrderCat": [self.PreferedOrderCat],
                "MaritalStatus": [self.MaritalStatus],
                "high_price": [self.high_price],
            }

            df = pd.DataFrame(data_dict)
            logger.info("DataFrame created successfully")

            return df

        except Exception as e:
            logger.error(f"Error while creating DataFrame: {e}")
            raise CustomException(e, sys)