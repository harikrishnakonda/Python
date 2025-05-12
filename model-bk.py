import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import lightgbm as lgb
import joblib
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder
import os

class RealEstateDataProcessor:
    def __init__(self, data_path):
        """Initialize the data processor"""
        self.data_path = data_path
        self.data = None
        self.location_stats = None
        self.location_price_per_sqft = None
        self.price_trends = None
        self.recent_years_weight = 3  # Give more weight to last 3 years
        self.exact_property_data = None
        
    def load_data(self):
        """Load data from CSV file"""
        print("\nLoading data...")
        self.data = pd.read_csv(self.data_path)
        print(f"Initial data shape: {self.data.shape}")
        return self
    
    def analyze_data(self):
        """Analyze data distributions and relationships"""
        print("\nData Analysis:")
        print(f"Total records: {len(self.data):,}")
        
        # Analyze basic statistics
        print("\nBasic Statistics:")
        for col in ['transaction_amount', 'procedure_area', 'beds', 'floor']:
            stats = self.data[col].describe()
            print(f"\n{col}:")
            print(stats)
        
        # Analyze price per square foot
        self.data['price_per_sqft'] = self.data['transaction_amount'] / self.data['procedure_area']
        print("\nPrice per square foot statistics:")
        print(self.data['price_per_sqft'].describe())
        
        # Analyze location distribution
        location_counts = self.data['location_leaf_name_en'].value_counts()
        print(f"\nNumber of unique locations: {len(location_counts)}")
        print("\nTop 10 locations by transaction count:")
        print(location_counts.head(10))
        
        # Analyze unit series distribution
        unit_counts = self.data['unit_series'].value_counts()
        print(f"\nNumber of unique unit series: {len(unit_counts)}")
        print("\nTop 10 unit series by transaction count:")
        print(unit_counts.head(10))
        
        return self
    
    def clean_data(self):
        """Clean and filter the data"""
        print("\nInitial data shape:", self.data.shape)
        
        # Convert date to datetime
        self.data['date_transaction_nk'] = pd.to_datetime(self.data['date_transaction_nk'])
        
        # Filter for recent data (2022-2025)
        self.data = self.data[self.data['date_transaction_nk'].dt.year >= 2022].copy()
        print("Data shape after year filtering:", self.data.shape)
        
        # Create a copy of the original data for exact property matches
        self.exact_property_data = self.data.copy()
        
        # Remove outliers
        self.data = self.data[
            (self.data['transaction_amount'] > 0) &
            (self.data['procedure_area'] > 0) &
            (self.data['beds'] >= 0)
        ]
        
        # Remove extreme outliers (outside 3 std from mean)
        for col in ['transaction_amount', 'procedure_area']:
            mean = self.data[col].mean()
            std = self.data[col].std()
            self.data = self.data[
                (self.data[col] >= mean - 3*std) &
                (self.data[col] <= mean + 3*std)
            ]
        
        print("Data shape after cleaning:", self.data.shape)
        
        return self
    
    def engineer_features(self):
        """Engineer new features"""
        print("\nEngineering features...")
        
        # Convert unit_series to numeric
        self.data['unit_series'] = pd.to_numeric(self.data['unit_series'], errors='coerce')
        
        # Calculate floor quartiles per location
        self.data['floor_q25'] = self.data.groupby('location_leaf_name_en')['floor'].transform(lambda x: x.quantile(0.25))
        self.data['floor_q75'] = self.data.groupby('location_leaf_name_en')['floor'].transform(lambda x: x.quantile(0.75))
        
        # Floor-related features
        self.data['is_low_floor'] = self.data['floor'] <= self.data['floor_q25']
        self.data['is_mid_floor'] = (self.data['floor'] > self.data['floor_q25']) & (self.data['floor'] <= self.data['floor_q75'])
        self.data['is_high_floor'] = self.data['floor'] > self.data['floor_q75']
        self.data['floor_ratio'] = self.data['floor'] / self.data.groupby('location_leaf_name_en')['floor'].transform('max')
        
        # Drop temporary columns
        self.data.drop(['floor_q25', 'floor_q75'], axis=1, inplace=True)
        
        # Price per square meter
        self.data['price_per_sqm'] = self.data['transaction_amount'] / self.data['procedure_area']
        
        # Time-based features
        current_date = pd.Timestamp.now()
        self.data['months_since_transaction'] = ((current_date - self.data['date_transaction_nk'])
                                               .dt.total_seconds() / (30 * 24 * 60 * 60))
        
        # Extract year, quarter, month
        self.data['transaction_year'] = self.data['date_transaction_nk'].dt.year
        self.data['transaction_quarter'] = self.data['date_transaction_nk'].dt.quarter
        self.data['transaction_month'] = self.data['date_transaction_nk'].dt.month
        
        # Area-related features
        self.data['area_log'] = np.log(self.data['procedure_area'])
        self.data['area_squared'] = self.data['procedure_area'] ** 2
        
        # Interaction features
        self.data['floor_x_area'] = self.data['floor'] * self.data['procedure_area']
        self.data['beds_x_area'] = self.data['beds'] * self.data['procedure_area']
        
        # Location features
        self.data['location_median_price'] = self.data.groupby('location_leaf_name_en')['transaction_amount'].transform('median')
        self.data['location_mean_price'] = self.data.groupby('location_leaf_name_en')['transaction_amount'].transform('mean')
        self.data['location_price_std'] = self.data.groupby('location_leaf_name_en')['transaction_amount'].transform('std')
        
        # Location area statistics
        self.data['location_median_area'] = self.data.groupby('location_leaf_name_en')['procedure_area'].transform('median')
        self.data['location_mean_area'] = self.data.groupby('location_leaf_name_en')['procedure_area'].transform('mean')
        
        # Location price per sqm statistics
        self.data['location_median_price_sqm'] = self.data.groupby('location_leaf_name_en')['price_per_sqm'].transform('median')
        self.data['location_mean_price_sqm'] = self.data.groupby('location_leaf_name_en')['price_per_sqm'].transform('mean')
        self.data['location_price_sqm_std'] = self.data.groupby('location_leaf_name_en')['price_per_sqm'].transform('std')
        
        # Price to location ratio
        self.data['price_to_location_ratio'] = self.data['transaction_amount'] / self.data['location_median_price']
        
        # Price segments
        self.data['price_segment'] = pd.qcut(self.data['transaction_amount'], q=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        
        # Recent transactions (last 6 months)
        recent_mask = self.data['months_since_transaction'] <= 6
        recent_groups = self.data[recent_mask].groupby('location_leaf_name_en')
        
        self.data['recent_median_price'] = self.data['location_leaf_name_en'].map(
            recent_groups['transaction_amount'].median()
        )
        self.data['recent_mean_price'] = self.data['location_leaf_name_en'].map(
            recent_groups['transaction_amount'].mean()
        )
        self.data['recent_price_sqm'] = self.data['location_leaf_name_en'].map(
            recent_groups['price_per_sqm'].median()
        )
        
        # Fill missing values with overall medians
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.data[numeric_cols] = self.data[numeric_cols].fillna(self.data[numeric_cols].median())
        
        print("Feature engineering completed")
        return self
        
    def prepare_model_data(self):
        """Prepare data for modeling"""
        print("\nPreparing model data...")
        
        # Calculate weights based on recency
        max_date = self.data['date_transaction_nk'].max()
        self.data['days_from_max'] = (max_date - self.data['date_transaction_nk']).dt.days
        self.data['sample_weight'] = np.exp(-self.data['days_from_max'] / 365)  # Exponential decay with 1-year half-life
        
        # Select features for modeling
        feature_cols = [
            # Basic features
            'floor', 'beds', 'procedure_area', 'has_parking',
            
            # Time features
            'months_since_transaction', 'transaction_year', 'transaction_quarter',
            'transaction_month',
            
            # Area features
            'area_log', 'area_squared',
            
            # Floor features
            'is_low_floor', 'is_mid_floor', 'is_high_floor', 'floor_ratio',
            
            # Price features
            'price_per_sqm', 'price_segment',
            
            # Location features
            'location_median_price_sqm', 'location_mean_price_sqm', 'location_price_sqm_std',
            'location_median_price', 'location_mean_price', 'location_price_std',
            'location_median_area', 'location_mean_area',
            
            # Interaction features
            'floor_x_area', 'beds_x_area', 'price_to_location_ratio',
            
            # Recent transaction features
            'recent_median_price', 'recent_mean_price', 'recent_price_sqm',
            
            # Categorical features
            'location_leaf_name_en', 'property_completion_status_sk', 'unit_series',
            'nearest_landmark_en', 'nearest_metro_en', 'nearest_mall_en'
        ]
        
        # Prepare features
        X = self.data[feature_cols].copy()
        
        # Prepare target (log of price)
        y = np.log(self.data['transaction_amount'])
        
        # Get sample weights
        weights = self.data['sample_weight']
        
        print("\nModel data preparation completed:")
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        return X, y, weights

    def get_location_data(self, location):
        """Get data for a specific location"""
        location_data = self.data[
            self.data['location_leaf_name_en'].str.lower() == location.lower()
        ].copy()
        
        # Add exact property data for this location
        exact_location_data = self.exact_property_data[
            self.exact_property_data['location_leaf_name_en'].str.lower() == location.lower()
        ]
        
        # Combine the data, keeping all exact property data
        location_data = pd.concat([location_data, exact_location_data]).drop_duplicates()
        
        print(f"\nFound {len(location_data)} transactions in {location}")
        return location_data

class RealEstateModel:
    def __init__(self):
        # Define feature groups
        self.numeric_features = [
            'floor', 'beds', 'procedure_area', 'months_since_transaction',
            'transaction_year', 'transaction_quarter', 'transaction_month',
            'area_log', 'area_squared', 'floor_ratio', 'price_per_sqm',
            'location_median_price', 'location_mean_price', 'location_price_std',
            'location_median_area', 'location_mean_area',
            'location_median_price_sqm', 'location_mean_price_sqm', 'location_price_sqm_std',
            'floor_x_area', 'beds_x_area', 'price_to_location_ratio',
            'recent_median_price', 'recent_mean_price', 'recent_price_sqm',
            'unit_series'  
        ]
        
        # Features for target encoding (high cardinality)
        self.target_features = [
            'location_leaf_name_en'  
        ]
        
        # Features for one-hot encoding (small cardinality)
        self.onehot_features = [
            'property_completion_status_sk',
            'nearest_landmark_en', 
            'nearest_metro_en', 
            'nearest_mall_en',
            'is_low_floor', 
            'is_mid_floor', 
            'is_high_floor', 
            'price_segment'
        ]
        
        # Create preprocessing steps
        numeric_transformer = Pipeline(steps=[
            ('scaler', RobustScaler())
        ])
        
        # Target encoder for high-cardinality features
        target_transformer = Pipeline(steps=[
            ('encoder', TargetEncoder(min_samples_leaf=5))  
        ])
        
        # One-hot encoder for low-cardinality features
        onehot_transformer = Pipeline(steps=[
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('target', target_transformer, self.target_features),
                ('onehot', onehot_transformer, self.onehot_features)
            ])
        
        # Create full pipeline with LightGBM
        self.pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', lgb.LGBMRegressor(
                n_estimators=200,  
                learning_rate=0.1,
                num_leaves=31,
                max_depth=8,
                min_child_samples=10,  
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ))
        ])
        
    def fit(self, X, y, weights):
        """Train the model"""
        print("\nTraining model...")
        self.model = self.pipeline.fit(X, y, regressor__sample_weight=weights)
        return self
        
    def predict(self, X):
        """Make predictions"""
        return self.pipeline.predict(X)
        
    def evaluate(self, X, y):
        """Evaluate model performance"""
        predictions = self.predict(X)
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        
        print("\nModel Performance:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        # Calculate MAPE
        y_true = np.exp(y)
        y_pred = np.exp(predictions)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        print(f"Mean Absolute Percentage Error: {mape:.2f}%")
        
        return {'mse': mse, 'r2': r2, 'mape': mape}
        
    def save_model(self, filename='real_estate_model.joblib'):
        """Save the model to a file"""
        joblib.dump(self.pipeline, filename)
        print(f"\nModel saved to {filename}")
        
    def load_model(self, filename='real_estate_model.joblib'):
        """Load the model from a file"""
        if os.path.exists(filename):
            self.pipeline = joblib.load(filename)
            print(f"\nModel loaded from {filename}")
        else:
            print(f"\nModel file {filename} not found. Training a new model...")
            self.train_new_model()
            
    def train_new_model(self):
        processor = RealEstateDataProcessor('sales_transactions.csv')
        processor.load_data()
        processor.clean_data()
        processor.engineer_features()
        X, y, weights = processor.prepare_model_data()
        
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
            X, y, weights, test_size=0.2, random_state=42
        )
        
        self.fit(X_train, y_train, weights_train)
        self.evaluate(X_test, y_test)
        self.save_model()

def evaluate_model_fit(X, y, model):
    """Evaluate if model is overfitting or underfitting using cross-validation and learning curves"""
    from sklearn.model_selection import cross_val_score, learning_curve
    import numpy as np
    
    print("\nChecking for Overfitting/Underfitting:")
    print("-" * 50)
    
    # Perform k-fold cross-validation
    print("\n1. Cross-validation Scores:")
    cv_scores = cross_val_score(model.pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores)
    
    print(f"Cross-validation RMSE scores:")
    for i, score in enumerate(cv_rmse, 1):
        print(f"Fold {i}: {score:.4f}")
    print(f"Mean RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std() * 2:.4f})")
    
    # Generate learning curves
    print("\n2. Learning Curve Analysis:")
    train_sizes = np.linspace(0.1, 1.0, 5)
    train_sizes, train_scores, val_scores = learning_curve(
        model.pipeline, X, y,
        train_sizes=train_sizes,
        cv=5,
        scoring='neg_mean_squared_error'
    )
    
    train_rmse = np.sqrt(-train_scores)
    val_rmse = np.sqrt(-val_scores)
    
    print("\nLearning Curve Results:")
    print("Training Set Size | Training RMSE | Validation RMSE")
    print("-" * 50)
    for size, train_score, val_score in zip(train_sizes, train_rmse.mean(axis=1), val_rmse.mean(axis=1)):
        print(f"{size:14.0f} | {train_score:12.4f} | {val_score:14.4f}")
    
    # Analyze results
    print("\n3. Model Fit Analysis:")
    
    # Check for high bias (underfitting)
    final_train_score = train_rmse.mean(axis=1)[-1]
    if final_train_score > 0.1:  
        print("⚠️ Potential Underfitting: High training error suggests the model might be too simple")
    
    # Check for high variance (overfitting)
    train_val_gap = abs(train_rmse.mean(axis=1)[-1] - val_rmse.mean(axis=1)[-1])
    if train_val_gap > 0.05:  
        print("⚠️ Potential Overfitting: Large gap between training and validation scores")
    
    # Check for good fit
    if final_train_score <= 0.1 and train_val_gap <= 0.05:
        print("✅ Good Fit: Model shows balanced performance between training and validation sets")
    
    return {
        'cv_rmse_mean': cv_rmse.mean(),
        'cv_rmse_std': cv_rmse.std(),
        'final_train_rmse': train_rmse.mean(axis=1)[-1],
        'final_val_rmse': val_rmse.mean(axis=1)[-1]
    }

def evaluate_model_performance(model, X_test, y_test):
    """
    Evaluate model performance using actual vs predicted prices
    
    Args:
        model: Trained model instance
        X_test: Test features
        y_test: Test target values (log prices)
    
    Returns:
        Dictionary containing performance metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Convert log prices back to actual prices
    y_true = np.exp(y_test)
    y_pred = np.exp(y_pred)
    
    print(f"\nEvaluating model performance...")
    print(f"Total samples: {len(y_true)}")
    
    # Calculate metrics
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    print("\nModel Performance Metrics:")
    print(f"R² Score: {r2:.4f}")
    print(f"MAPE: {mape*100:.2f}%")
    print(f"MAE: ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    
    # Print detailed analysis
    print("\nDetailed Analysis:")
    print(f"Average Actual Price: ${y_true.mean():,.2f}")
    print(f"Average Predicted Price: ${y_pred.mean():,.2f}")
    print(f"Price Range: ${y_true.min():,.2f} to ${y_true.max():,.2f}")
    
    # Calculate prediction accuracy bands
    within_5_percent = np.mean(np.abs((y_pred - y_true) / y_true) <= 0.05) * 100
    within_10_percent = np.mean(np.abs((y_pred - y_true) / y_true) <= 0.10) * 100
    within_15_percent = np.mean(np.abs((y_pred - y_true) / y_true) <= 0.15) * 100
    
    print("\nPrediction Accuracy:")
    print(f"Within 5% of actual price: {within_5_percent:.1f}%")
    print(f"Within 10% of actual price: {within_10_percent:.1f}%")
    print(f"Within 15% of actual price: {within_15_percent:.1f}%")
    
    return {
        'r2': r2,
        'mape': mape,
        'mae': mae,
        'rmse': rmse,
        'within_5_percent': within_5_percent,
        'within_10_percent': within_10_percent,
        'within_15_percent': within_15_percent
    }

def train_and_evaluate():
    """Train the model and evaluate its performance"""
    # Load and process data
    processor = RealEstateDataProcessor('sales_transactions.csv')
    processor.load_data()
    processor.clean_data()
    processor.engineer_features()
    X, y, weights = processor.prepare_model_data()
    
    # Split data
    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42
    )
    
    # Initialize and train model
    print("\nTraining model...")
    model = RealEstateModel()
    model.fit(X_train, y_train, weights_train)
    
    # Check for overfitting/underfitting
    print("\nChecking for overfitting/underfitting...")
    fit_metrics = evaluate_model_fit(X_train, y_train, model)
    
    # Evaluate model performance
    print("\nEvaluating model performance...")
    metrics = evaluate_model_performance(model, X_test, y_test)
    
    # Save model
    print("\nSaving model...")
    model.save_model()
    print("Model saved successfully!")
    
    return metrics, fit_metrics

def analyze_floor_data(location_data):
    """Analyze floor-specific transaction data"""
    floor_data = []
    for floor_num in sorted(location_data['floor'].unique()):
        floor_transactions = location_data[location_data['floor'] == floor_num]
        if len(floor_transactions) > 0:
            latest_transaction = floor_transactions.loc[floor_transactions['date_transaction_nk'].idxmax()]
            floor_data.append({
                'floor': int(floor_num),
                'count': len(floor_transactions),
                'recent_price': int(latest_transaction['transaction_amount']),
                'price_sqm': int(latest_transaction['transaction_amount'] / latest_transaction['procedure_area']),
                'last_trans': latest_transaction['date_transaction_nk'].strftime('%Y-%m')
            })
    return floor_data

def calculate_location_floor_weights(location_data):
    """Calculate floor-specific price weights"""
    # Calculate price per sqm for each floor
    floor_prices = []
    for floor in sorted(location_data['floor'].unique()):
        floor_data = location_data[location_data['floor'] == floor]
        if len(floor_data) > 0:
            price_per_sqm = floor_data['transaction_amount'].median() / floor_data['procedure_area'].median()
            floor_prices.append((floor, price_per_sqm))
    
    if len(floor_prices) < 2:
        return {'premium': 1.0}
    
    # Calculate floor premium based on price trends
    floor_prices.sort(key=lambda x: x[0])  # Sort by floor number
    floor_premiums = []
    
    for i in range(1, len(floor_prices)):
        if floor_prices[i][1] > 0 and floor_prices[i-1][1] > 0:
            premium = floor_prices[i][1] / floor_prices[i-1][1]
            if 0.8 <= premium <= 1.2:  # Filter out extreme values
                floor_premiums.append(premium)
    
    if len(floor_premiums) > 0:
        avg_premium = sum(floor_premiums) / len(floor_premiums)
        return {'premium': avg_premium}
    
    return {'premium': 1.0}

def get_large_unit_series(building_data, area_threshold=200):
    """Determine which unit series are typically larger based on historical data"""
    # Group by unit series and calculate median area
    series_stats = building_data.groupby('unit_series')['procedure_area'].agg(['median', 'count']).reset_index()
    
    # Consider series with at least 2 transactions to be more reliable
    reliable_series = series_stats[series_stats['count'] >= 2]
    
    # Identify series that typically have larger areas
    large_series = reliable_series[reliable_series['median'] > area_threshold]['unit_series'].tolist()
    
    return large_series

def get_most_common_beds(units, location_data):
    """Get the most common bed count for a given set of units"""
    beds = units['beds'].mode().iloc[0] if len(units['beds'].mode()) > 0 else 2
    return beds

def get_property_details(location_data, property_number):
    """Extract property details from historical data based on property number"""
    # Parse floor and unit series
    floor, unit_series = parse_property_number(property_number)
    
    # First check if we have exact property number matches
    exact_matches = location_data[location_data['property_number'].astype(str) == str(property_number)]
    if len(exact_matches) > 0:
        # Use the exact configuration from data
        beds = exact_matches['beds'].iloc[0]
        area = exact_matches['procedure_area'].iloc[0]
        print(f"\nFound exact match for property {property_number}:")
        print(f"Beds: {beds}")
        print(f"Area: {area:.2f} sqm")
        print(f"Total matches: {len(exact_matches)}")
        return {
            'floor': floor,
            'unit_series': unit_series,
            'beds': beds,
            'area': area
        }
    
    # If no exact match, look for similar units
    similar_units = location_data[
        (location_data['floor'] == floor) & 
        (location_data['unit_series'] == unit_series)
    ]
    
    if len(similar_units) > 0:
        # Use the most common configuration for this unit type
        beds = similar_units['beds'].iloc[0]
        area = similar_units['procedure_area'].median()
    else:
        # If no exact matches, look for units with same series on other floors
        series_units = location_data[location_data['unit_series'] == unit_series]
        if len(series_units) > 0:
            beds = series_units['beds'].iloc[0]
            area = series_units['procedure_area'].median()
        else:
            # If still no matches, use floor position to estimate
            # Calculate median areas for each unit series
            series_stats = location_data.groupby('unit_series').agg({
                'procedure_area': ['median', 'count'],
                'beds': 'first'
            }).round(2)
            
            # Find similar unit series based on position
            unit_series_num = int(unit_series)
            closest_series = min(
                [int(s) for s in location_data['unit_series'].unique()],
                key=lambda x: abs(x - unit_series_num)
            )
            closest_series = f"{closest_series:02d}"
            
            if closest_series in series_stats.index:
                beds = series_stats.loc[closest_series, ('beds', 'first')]
                area = series_stats.loc[closest_series, ('procedure_area', 'median')]
            else:
                # Use overall medians as last resort
                beds = location_data['beds'].median()
                area = location_data['procedure_area'].median()
    
    print(f"\nProperty configuration analysis:")
    print(f"Unit series distribution:")
    series_stats = location_data.groupby('unit_series').agg({
        'procedure_area': ['median', 'count'],
        'beds': ['min', 'max']
    }).round(2)
    print(series_stats)
    
    return {
        'floor': floor,
        'unit_series': unit_series,
        'beds': beds,
        'area': area
    }

def parse_property_number(property_number):
    """Parse property number to get floor and unit series
    Example: 1108 -> floor: 11, unit_series: 08"""
    if not property_number or len(str(property_number)) != 4:
        return None, None
    
    property_number = str(property_number)
    floor = int(property_number[:2])
    unit_series = property_number[2:]
    return floor, unit_series

def prepare_sample_property(location_data, location, beds, area, floor, unit_series):
    """Prepare a sample property with all required features"""
    current_date = pd.Timestamp.now()
    
    # Basic features
    sample_property = pd.DataFrame({
        'location_leaf_name_en': [location],
        'beds': [beds],
        'procedure_area': [area],
        'floor': [floor if floor is not None else 0],
        'unit_series': [int(unit_series) if unit_series is not None else 0],
        'date_transaction_nk': [current_date]
    })
    
    # Time-based features
    sample_property['months_since_transaction'] = 0
    sample_property['transaction_year'] = current_date.year
    sample_property['transaction_quarter'] = current_date.quarter
    sample_property['transaction_month'] = current_date.month
    
    # Area-related features
    sample_property['area_log'] = np.log(sample_property['procedure_area'])
    sample_property['area_squared'] = sample_property['procedure_area'] ** 2
    
    # Floor-related features
    floor_q25 = location_data['floor'].quantile(0.25)
    floor_q75 = location_data['floor'].quantile(0.75)
    sample_property['is_low_floor'] = sample_property['floor'] <= floor_q25
    sample_property['is_mid_floor'] = (sample_property['floor'] > floor_q25) & (sample_property['floor'] <= floor_q75)
    sample_property['is_high_floor'] = sample_property['floor'] > floor_q75
    sample_property['floor_ratio'] = sample_property['floor'] / location_data['floor'].max()
    
    # Price per sqm features
    location_data['price_per_sqm'] = location_data['transaction_amount'] / location_data['procedure_area']
    sample_property['price_per_sqm'] = location_data['price_per_sqm'].median()
    
    # Location price statistics
    sample_property['location_median_price'] = location_data['transaction_amount'].median()
    sample_property['location_mean_price'] = location_data['transaction_amount'].mean()
    sample_property['location_price_std'] = location_data['transaction_amount'].std()
    
    # Location area statistics
    sample_property['location_median_area'] = location_data['procedure_area'].median()
    sample_property['location_mean_area'] = location_data['procedure_area'].mean()
    
    # Location price per sqm statistics
    sample_property['location_median_price_sqm'] = location_data['price_per_sqm'].median()
    sample_property['location_mean_price_sqm'] = location_data['price_per_sqm'].mean()
    sample_property['location_price_sqm_std'] = location_data['price_per_sqm'].std()
    
    # Interaction features
    sample_property['floor_x_area'] = sample_property['floor'] * sample_property['procedure_area']
    sample_property['beds_x_area'] = sample_property['beds'] * sample_property['procedure_area']
    sample_property['price_to_location_ratio'] = 1.0
    
    # Recent transaction features
    recent_data = location_data[
        location_data['date_transaction_nk'] >= current_date - pd.DateOffset(months=6)
    ]
    if len(recent_data) > 0:
        sample_property['recent_median_price'] = recent_data['transaction_amount'].median()
        sample_property['recent_mean_price'] = recent_data['transaction_amount'].mean()
        sample_property['recent_price_sqm'] = recent_data['price_per_sqm'].median()
    else:
        sample_property['recent_median_price'] = sample_property['location_median_price']
        sample_property['recent_mean_price'] = sample_property['location_mean_price']
        sample_property['recent_price_sqm'] = sample_property['location_median_price_sqm']
    
    # Categorical features
    sample_property['property_completion_status_sk'] = 'completed'
    sample_property['nearest_landmark_en'] = location_data['nearest_landmark_en'].mode().iloc[0]
    sample_property['nearest_metro_en'] = location_data['nearest_metro_en'].mode().iloc[0]
    sample_property['nearest_mall_en'] = location_data['nearest_mall_en'].mode().iloc[0]
    
    # Price segment (use median as default)
    sample_property['price_segment'] = 'medium'
    
    return sample_property

def get_location_factors(location_data):
    """Calculate location-specific factors based on historical data"""
    factors = {}
    
    # Calculate median price per sqm for each location
    location_stats = location_data.groupby('location_l3_id').agg({
        'transaction_amount': ['median', 'count'],
        'procedure_area': 'median'
    }).reset_index()
    
    location_stats.columns = ['location_l3_id', 'median_price', 'transaction_count', 'median_area']
    location_stats['price_per_sqm'] = location_stats['median_price'] / location_stats['median_area']
    
    # Calculate relative price factors
    avg_price_per_sqm = location_stats['price_per_sqm'].median()
    for _, row in location_stats.iterrows():
        if row['transaction_count'] >= 3:  # Only use locations with sufficient data
            factors[row['location_l3_id']] = {
                'price_factor': row['price_per_sqm'] / avg_price_per_sqm,
                'transaction_count': row['transaction_count'],
                'median_price': row['median_price'],
                'median_area': row['median_area']
            }
    
    return factors

def calculate_base_price(location_data, area, beds, location_l3_id=None):
    """Calculate base price from historical data based on property characteristics"""
    # First try to get recent transactions (last 12 months) for similar units
    recent_cutoff = pd.Timestamp.now() - pd.DateOffset(months=12)
    recent_data = location_data[location_data['date_transaction_nk'] >= recent_cutoff]
    
    similar_units = recent_data[
        (recent_data['beds'] == beds) &
        (recent_data['procedure_area'].between(area * 0.9, area * 1.1))
    ]
    
    if len(similar_units) > 0:
        # Use median of recent similar units
        base_price = similar_units['transaction_amount'].median()
        print(f"\nBase price from recent similar units: ${base_price:,.2f}")
        return base_price
    
    # If no recent similar units, look at all similar units
    similar_units = location_data[
        (location_data['beds'] == beds) &
        (location_data['procedure_area'].between(area * 0.9, area * 1.1))
    ]
    
    if len(similar_units) > 0:
        base_price = similar_units['transaction_amount'].median()
        print(f"\nBase price from all similar units: ${base_price:,.2f}")
        return base_price
    
    # If still no matches, use price per sqm approach
    recent_comparable = recent_data[recent_data['procedure_area'].between(area * 0.7, area * 1.3)]
    if len(recent_comparable) > 0:
        median_price_per_sqm = (recent_comparable['transaction_amount'] / recent_comparable['procedure_area']).median()
        base_price = median_price_per_sqm * area
        print(f"\nBase price from recent comparable units: ${base_price:,.2f}")
        return base_price
    
    # Final fallback: use overall median price per sqm
    price_per_sqm = location_data['transaction_amount'] / location_data['procedure_area']
    median_price_per_sqm = price_per_sqm.median()
    base_price = median_price_per_sqm * area
    print(f"\nBase price from overall median: ${base_price:,.2f}")
    return base_price

def predict_property_price(location, property_number):
    """Predict property price based on location and property number"""
    # Load and prepare data
    processor = RealEstateDataProcessor('sales_transactions.csv')
    processor.load_data()
    processor.clean_data()
    
    # Get location-specific data
    location_data = processor.get_location_data(location)
    location_l3_id = location_data['location_l3_id'].iloc[0] if len(location_data) > 0 else None
    
    print(f"\nFound {len(location_data)} transactions in {location}")
    if location_l3_id:
        print(f"Location L3 ID: {location_l3_id}")
    
    # Get property details
    property_details = get_property_details(location_data, property_number)
    floor = property_details['floor']
    unit_series = property_details['unit_series']
    beds = property_details['beds']
    area = property_details['area']
    
    print(f"\nProperty details from number {property_number}:")
    print(f"Floor: {floor}")
    print(f"Unit Series: {unit_series}")
    print(f"Bedrooms: {beds}")
    print(f"Area: {area:.1f} sqm")
    
    # Analyze floor-specific data
    floor_data = analyze_floor_data(location_data)
    print("\nFloor-specific Analysis (Most Recent Transaction per Floor):")
    print("Floor  Count  RecentPrice  PriceSqm  LastTrans")
    print("-" * 50)
    for f_data in floor_data:
        print(f"{f_data['floor']:2d}      {f_data['count']:d}    ${f_data['recent_price']:,}   ${f_data['price_sqm']:,}   {f_data['last_trans']}")
    
    # Calculate floor premium
    floor_weights = calculate_location_floor_weights(location_data)
    floor_premium = 1.0
    
    # Check if this is a high floor
    max_floor = max(location_data['floor'])
    if floor > max_floor * 0.8:  # Top 20% of floors
        high_floor_bonus = 1.06  # Increased from 1.04
        print(f"\nFloor premium factor: {floor_weights['premium']:.2f}x")
        print(f"High floor bonus: {high_floor_bonus:.2f}x")
        floor_premium = floor_weights['premium'] * high_floor_bonus
    else:
        floor_premium = floor_weights['premium']
        print(f"\nFloor premium factor: {floor_premium:.2f}x")
    
    # Calculate base price
    base_price = calculate_base_price(location_data, area, beds, location_l3_id)
    
    # Apply floor premium
    final_price = base_price * floor_premium
    
    # Check for recent transactions of the exact unit
    exact_unit = location_data[location_data['property_number'].astype(str) == str(property_number)]
    if len(exact_unit) > 0:
        recent_price = exact_unit['transaction_amount'].iloc[0]
        # Weight recent transaction more heavily (80/20 instead of 70/30)
        final_price = (recent_price * 0.8) + (final_price * 0.2)
    
    # Apply market trend adjustment
    recent_cutoff = pd.Timestamp.now() - pd.DateOffset(months=6)
    recent_trend = location_data[location_data['date_transaction_nk'] >= recent_cutoff]
    if len(recent_trend) >= 3:
        recent_avg = recent_trend['transaction_amount'].mean()
        older_avg = location_data[location_data['date_transaction_nk'] < recent_cutoff]['transaction_amount'].mean()
        if older_avg > 0:
            trend_factor = recent_avg / older_avg
            trend_factor = min(max(trend_factor, 0.9), 1.2)  # Cap between 0.9 and 1.2
            final_price *= trend_factor
            print(f"Market trend factor: {trend_factor:.2f}x")
    
    # Format output
    price_per_sqm = final_price / area
    
    print(f"\nPrediction for property in {location}:")
    print(f"Specifications: {beds} beds, {area:.1f} sqm")
    print(f"Property number: {property_number} (Floor {floor}, Unit {unit_series})")
    print(f"\nFinal prediction: ${final_price:,.2f}")
    print(f"Price per sqm: ${price_per_sqm:,.2f}")
    print(f"\nAdjustment factors applied:")
    print(f"Floor premium: {floor_premium:.2f}x")
    
    return final_price

def analyze_price_trends(location, beds, area):
    """Analyze price trends and model behavior"""
    processor = RealEstateDataProcessor('sales_transactions.csv')
    processor.load_data()
    
    # Convert date to datetime
    processor.data['date_transaction_nk'] = pd.to_datetime(processor.data['date_transaction_nk'])
    
    # Get location data
    location_data = processor.get_location_data(location)
    
    # Calculate price per sqm
    location_data['price_per_sqm'] = location_data['transaction_amount'] / location_data['procedure_area']
    
    # Sort by date
    location_data = location_data.sort_values('date_transaction_nk')
    
    print("\nPrice Trend Analysis:")
    print("-" * 50)
    
    # Overall price trends
    print("\n1. Overall Price Trends in", location)
    yearly_stats = location_data.groupby(location_data['date_transaction_nk'].dt.year).agg({
        'transaction_amount': ['mean', 'median', 'count'],
        'price_per_sqm': ['mean', 'median']
    }).round(2)
    print(yearly_stats)
    
    # Similar properties analysis
    similar_props = location_data[
        (location_data['beds'] == beds) &
        (location_data['procedure_area'].between(area * 0.8, area * 1.2))
    ]
    
    print(f"\n2. Similar Properties Analysis (2 bed, ~{area} sqm)")
    print(f"Total similar properties: {len(similar_props)}")
    
    if len(similar_props) > 0:
        # Price distribution
        percentiles = similar_props['transaction_amount'].quantile([0.25, 0.5, 0.75])
        print("\nPrice Distribution:")
        print(f"25th percentile: ${percentiles[0.25]:,.2f}")
        print(f"Median: ${percentiles[0.5]:,.2f}")
        print(f"75th percentile: ${percentiles[0.75]:,.2f}")
        
        # Recent vs Old transactions
        recent_cutoff = similar_props['date_transaction_nk'].max() - pd.DateOffset(months=12)
        recent_props = similar_props[similar_props['date_transaction_nk'] >= recent_cutoff]
        old_props = similar_props[similar_props['date_transaction_nk'] < recent_cutoff]
        
        print("\nPrice Comparison:")
        if len(recent_props) > 0:
            print(f"Recent 12 months median: ${recent_props['transaction_amount'].median():,.2f}")
            print(f"Recent 12 months mean: ${recent_props['transaction_amount'].mean():,.2f}")
        if len(old_props) > 0:
            print(f"Previous periods median: ${old_props['transaction_amount'].median():,.2f}")
            print(f"Previous periods mean: ${old_props['transaction_amount'].mean():,.2f}")
        
        # Floor analysis
        print("\nPrice by Floor Range:")
        similar_props['floor_range'] = pd.cut(similar_props['floor'], 
                                            bins=[0, 10, 20, 30, 100],
                                            labels=['1-10', '11-20', '21-30', '31+'])
        floor_stats = similar_props.groupby('floor_range')['transaction_amount'].agg(['median', 'count'])
        print(floor_stats)
        
        # Most recent transactions
        print("\nMost Recent Transactions:")
        recent = similar_props.nlargest(5, 'date_transaction_nk')
        for _, row in recent.iterrows():
            print(f"Date: {row['date_transaction_nk'].strftime('%Y-%m-%d')}")
            print(f"Price: ${row['transaction_amount']:,.2f}")
            print(f"Floor: {row['floor']}")
            print(f"Price/sqm: ${row['price_per_sqm']:,.2f}")
            print("---")

def calculate_growth_factor(location_data):
    """Calculate price growth factor based on historical data"""
    # Calculate quarterly median prices for recent data (last 2 years)
    recent_cutoff = pd.Timestamp.now() - pd.DateOffset(years=2)
    recent_data = location_data[location_data['date_transaction_nk'] >= recent_cutoff]
    
    quarterly_prices = recent_data.groupby(
        [recent_data['date_transaction_nk'].dt.year.rename('year'),
         recent_data['date_transaction_nk'].dt.quarter.rename('quarter')]
    ).agg({
        'transaction_amount': 'median',
        'procedure_area': 'median',
        'transaction_amount': 'count'
    }).reset_index()
    
    # Calculate price per sqm
    quarterly_prices['price_per_sqm'] = quarterly_prices['transaction_amount'] / quarterly_prices['procedure_area']
    
    # Only consider quarters with enough data
    quarterly_prices = quarterly_prices[quarterly_prices['transaction_amount'] >= 2]
    
    if len(quarterly_prices) > 1:
        # Sort by year and quarter
        quarterly_prices = quarterly_prices.sort_values(
            by=['year', 'quarter']
        )
        
        # Calculate quarter-over-quarter growth rates
        quarterly_prices['growth_rate'] = quarterly_prices['price_per_sqm'].pct_change()
        
        # Use recent growth rates (last 2 quarters if available)
        recent_growth = quarterly_prices['growth_rate'].tail(2).mean()
        
        # If growth rate is available, return annualized factor
        if pd.notnull(recent_growth) and recent_growth > -0.5:  # Sanity check on growth rate
            annualized_growth = (1 + recent_growth) ** 8  # Annualized from quarterly
            return min(max(annualized_growth, 1.0), 2.0)  # Clip between 1.0x and 2.0x
    
    return 1.0  # Default to no growth if not enough data

if __name__ == '__main__':
    print("\nTraining and evaluating model...")
    metrics, fit_metrics = train_and_evaluate()
    
    print("\nTesting specific properties:")
    print("\nTesting property 1108:")
    predict_property_price('sadaf 1', 1108)
    
    print("\nTesting property 1209:")
    predict_property_price('sadaf 1', 1209)
    
    print("\nTesting property 3901:")
    predict_property_price('sadaf 1', 3901)
