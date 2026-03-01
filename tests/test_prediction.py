from src.pipeline.predict_pipeline import PredictPipeline, CustomData

custom_data = CustomData(
    CityTier=1,
    WarehouseToHome=10,
    HourSpendOnApp=3,
    NumberOfDeviceRegistered=2,
    SatisfactionScore=4,
    NumberOfAddress=1,
    Complain=0,
    OrderAmountHikeFromlastYear=15,
    CouponUsed=1,
    OrderCount=5,
    DaySinceLastOrder=2,
    CashbackAmount=20,
    Gender="Male",
    PreferredLoginDevice="Mobile",
    PreferredPaymentMode="COD",
    PreferedOrderCat="Laptop",
    MaritalStatus="Single",
    Tenure=12,
    high_price=1
)

df = custom_data.get_data_as_data_frame()

pipeline = PredictPipeline()
prediction = pipeline.predict(df)

print("Prediction:", prediction)
