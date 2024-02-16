from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier

class MachineLearning:
    def __init__(self, data_to_use, data_to_target, test_size=0.30, random_state=32):

        self.data_to_use = data_to_use
        self.data_to_target = data_to_target
        self.test_size = test_size
        self.random_state = random_state

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.knn_model = None

        self.decision_tree_gini_model = None
        self.decision_tree_depth_model = None
        self.decision_tree_entropy_model = None

        self.mae_train = None
        self.rmse_train = None
        self.r2_train = None
        self.mse_train = None


    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data_to_use,
            self.data_to_target,
            test_size=self.test_size,
            random_state=self.random_state
        )

    def fit_knn_classification(self):
        knn_model = KNeighborsClassifier(n_neighbors=5)
        #at this point it just saves the data and do no training
        self.knn_model = knn_model.fit(self.X_train, self.y_train)

    def predict_model(self, model):
        y_pred_knn = model.predict(self.X_test)
        return y_pred_knn

    def evaluate_model(self, y_test_pred):
        self.mae_train = mean_absolute_error(self.y_test, y_test_pred)
        self.rmse_train = mean_squared_error(self.y_test, y_test_pred, squared=False)
        self.r2_train = r2_score(self.y_test, y_test_pred)
        self.mse_train = mean_squared_error(self.y_test, y_test_pred)

    def decision_tree_gini_classifier(self):
        decision_gini = DecisionTreeClassifier()
        self.decision_tree_gini_model = decision_gini.fit(self.X_train, self.y_train)

    def decision_tree_depth_classifier(self):
        decision_depth = DecisionTreeClassifier(max_depth=5)
        self.decision_tree_depth_model = decision_depth.fit(self.X_train, self.y_train)

    def decision_tree_entropy_classifier(self):
        decision_entropy = DecisionTreeClassifier(criterion='entropy')
        self.decision_tree_entropy_model = decision_entropy.fit(self.X_train, self.y_train)
