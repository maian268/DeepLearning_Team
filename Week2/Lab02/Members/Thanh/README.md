Data/ chứa train.csv, test.csv, sample_submission.csv — input chung cho cả 3 model.

Notebooks/ đọc dữ liệu đó rồi tinh chỉnh từng model: Model1_XGBoost.ipynb, Model2_DecisionTree.ipynb, Model3_RandomForest.ipynb. Mỗi notebook preprocess, RandomizedSearchCV, predict, rồi ghi kết quả sang experiments/.

experiments/ lưu output theo model và thời gian chạy (config.json, metrics.json, model.pkl, predictions.csv): xgboost/, decision_tree/, random_forest/. Folder comparison/ là bảng tổng hợp sau khi so sánh.

SoSanh_CacModel.ipynb không train lại; chỉ đọc metrics.json trong experiments/ để xếp hạng RMSE.
