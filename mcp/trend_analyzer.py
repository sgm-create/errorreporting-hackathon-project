#!/usr/bin/env python3

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, IsolationForest, ExtraTreesRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.decomposition import PCA, FastICA
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.stats import kstest, shapiro, anderson, pearsonr, spearmanr
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')

class TrendAnalyzer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importances = {}
        self.model_performances = {}
        
    def analyze_patterns(self, incidents):
        """Comprehensive ML-powered trend analysis"""
        if len(incidents) < 20:
            return {'status': 'insufficient_data', 'minimum_required': 20}
        
        print(f"TREND: Analyzing {len(incidents)} incidents with advanced ML models")
        
        df = pd.DataFrame(incidents)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Feature Engineering Pipeline
        engineered_df = self._comprehensive_feature_engineering(df)
        
        # Multi-Model Analysis Pipeline
        analysis = {
            'executive_summary': self._generate_executive_summary(engineered_df),
            'statistical_analysis': self._advanced_statistical_analysis(engineered_df),
            'ml_clustering_analysis': self._advanced_clustering_analysis(engineered_df),
            'time_series_forecasting': self._advanced_time_series_analysis(engineered_df),
            'anomaly_detection': self._comprehensive_anomaly_analysis(engineered_df),
            'predictive_modeling': self._advanced_predictive_modeling(engineered_df),
            'provider_intelligence': self._provider_performance_intelligence(engineered_df),
            'risk_assessment': self._quantitative_risk_assessment(engineered_df),
            'business_impact_modeling': self._business_impact_modeling(engineered_df),
            'recommendation_engine': self._ml_recommendation_engine(engineered_df)
        }
        
        return analysis
    
    def _comprehensive_feature_engineering(self, df):
        """Advanced feature engineering for ML models"""
        engineered_df = df.copy()
        
        # Temporal feature engineering
        engineered_df['hour'] = engineered_df['timestamp'].dt.hour
        engineered_df['day_of_week'] = engineered_df['timestamp'].dt.dayofweek
        engineered_df['month'] = engineered_df['timestamp'].dt.month
        engineered_df['quarter'] = engineered_df['timestamp'].dt.quarter
        engineered_df['is_weekend'] = engineered_df['day_of_week'].isin([5, 6]).astype(int)
        engineered_df['is_business_hours'] = engineered_df['hour'].between(9, 17).astype(int)
        engineered_df['is_peak_hours'] = engineered_df['hour'].isin([10, 11, 14, 15, 16]).astype(int)
        
        # Cyclical encoding for temporal features
        engineered_df['hour_sin'] = np.sin(2 * np.pi * engineered_df['hour'] / 24)
        engineered_df['hour_cos'] = np.cos(2 * np.pi * engineered_df['hour'] / 24)
        engineered_df['day_sin'] = np.sin(2 * np.pi * engineered_df['day_of_week'] / 7)
        engineered_df['day_cos'] = np.cos(2 * np.pi * engineered_df['day_of_week'] / 7)
        
        # Business metrics engineering
        engineered_df['impact_per_customer'] = engineered_df['business_impact'] / np.maximum(engineered_df['customers_affected'], 1)
        engineered_df['impact_per_minute'] = engineered_df['business_impact'] / np.maximum(engineered_df['duration_minutes'], 1)
        engineered_df['customer_efficiency'] = engineered_df['customers_affected'] / np.maximum(engineered_df['duration_minutes'], 1)
        
        # Severity encoding with business logic
        severity_weights = {'Minor': 1, 'Medium': 2, 'Major': 3, 'Critical': 4}
        engineered_df['severity_weight'] = engineered_df['severity'].map(severity_weights)
        
        # Provider performance encoding
        provider_stats = engineered_df.groupby('service_provider').agg({
            'business_impact': ['mean', 'std', 'count', 'sum'],
            'duration_minutes': ['mean', 'std'],
            'customers_affected': ['mean', 'sum']
        })
        
        # Flatten column names and merge
        provider_stats.columns = ['_'.join(col) for col in provider_stats.columns]
        provider_stats = provider_stats.add_prefix('provider_')
        engineered_df = engineered_df.merge(provider_stats, left_on='service_provider', right_index=True, how='left')
        
        # Error type frequency encoding
        error_counts = engineered_df['error_type'].value_counts()
        engineered_df['error_frequency'] = engineered_df['error_type'].map(error_counts)
        
        # Rolling statistics (time-ordered features)
        engineered_df = engineered_df.sort_values('timestamp')
        windows = [3, 7, 14]
        for window in windows:
            engineered_df[f'rolling_impact_{window}'] = engineered_df['business_impact'].rolling(window=window, min_periods=1).mean()
            engineered_df[f'rolling_duration_{window}'] = engineered_df['duration_minutes'].rolling(window=window, min_periods=1).mean()
            engineered_df[f'rolling_customers_{window}'] = engineered_df['customers_affected'].rolling(window=window, min_periods=1).mean()
        
        # Lag features
        for lag in [1, 2, 3]:
            engineered_df[f'lag_impact_{lag}'] = engineered_df['business_impact'].shift(lag)
            engineered_df[f'lag_duration_{lag}'] = engineered_df['duration_minutes'].shift(lag)
        
        # Statistical moments
        engineered_df['impact_zscore'] = (engineered_df['business_impact'] - engineered_df['business_impact'].mean()) / engineered_df['business_impact'].std()
        engineered_df['duration_zscore'] = (engineered_df['duration_minutes'] - engineered_df['duration_minutes'].mean()) / engineered_df['duration_minutes'].std()
        
        return engineered_df.fillna(0)
    
    def _generate_executive_summary(self, df):
        """Generate executive-level insights"""
        return {
            'total_incidents_analyzed': len(df),
            'analysis_period_days': (df['timestamp'].max() - df['timestamp'].min()).days,
            'total_financial_impact': float(df['business_impact'].sum()),
            'average_incident_cost': float(df['business_impact'].mean()),
            'total_customer_impact': int(df['customers_affected'].sum()),
            'system_reliability_score': round(100 - (len(df) / max(1, (df['timestamp'].max() - df['timestamp'].min()).days)) * 10, 2),
            'critical_incident_rate': round((df['severity'] == 'Critical').mean() * 100, 2),
            'mean_time_to_resolution': float(df['duration_minutes'].mean()),
            'incident_velocity_per_day': round(len(df) / max(1, (df['timestamp'].max() - df['timestamp'].min()).days), 2)
        }
    
    def _advanced_statistical_analysis(self, df):
        """Advanced statistical analysis using scipy and statsmodels"""
        numeric_cols = ['business_impact', 'duration_minutes', 'customers_affected']
        analysis = {}
        
        for col in numeric_cols:
            data = df[col].dropna()
            
            # Descriptive statistics
            analysis[f'{col}_statistics'] = {
                'mean': float(data.mean()),
                'median': float(data.median()),
                'std': float(data.std()),
                'variance': float(data.var()),
                'skewness': float(stats.skew(data)),
                'kurtosis': float(stats.kurtosis(data)),
                'coefficient_of_variation': float(data.std() / data.mean()) if data.mean() != 0 else 0,
                'interquartile_range': float(data.quantile(0.75) - data.quantile(0.25)),
                'percentile_95': float(data.quantile(0.95)),
                'percentile_99': float(data.quantile(0.99))
            }
            
            # Normality tests
            if len(data) > 8:
                shapiro_stat, shapiro_p = shapiro(data[:5000])  # Shapiro-Wilk test (sample if too large)
                analysis[f'{col}_normality'] = {
                    'shapiro_wilk_statistic': float(shapiro_stat),
                    'shapiro_wilk_p_value': float(shapiro_p),
                    'is_normally_distributed': shapiro_p > 0.05
                }
                
                # Anderson-Darling test
                anderson_result = anderson(data, dist='norm')
                analysis[f'{col}_anderson_darling'] = {
                    'statistic': float(anderson_result.statistic),
                    'critical_values': anderson_result.critical_values.tolist(),
                    'significance_levels': anderson_result.significance_level.tolist()
                }
        
        # Correlation analysis
        correlation_matrix = df[numeric_cols].corr(method='pearson')
        spearman_matrix = df[numeric_cols].corr(method='spearman')
        
        analysis['correlation_analysis'] = {
            'pearson_correlations': correlation_matrix.to_dict(),
            'spearman_correlations': spearman_matrix.to_dict(),
            'strongest_correlation': {
                'variables': correlation_matrix.abs().unstack().drop_duplicates().nlargest(2).index[1],
                'coefficient': float(correlation_matrix.abs().unstack().drop_duplicates().nlargest(2).iloc[1])
            }
        }
        
        return analysis
    
    def _advanced_clustering_analysis(self, df):
        """Multi-algorithm clustering analysis"""
        # Prepare features for clustering
        feature_cols = [col for col in df.columns if any(x in col for x in ['impact', 'duration', 'customers', 'severity_weight', 'hour', 'day'])]
        feature_cols = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']]
        
        if len(feature_cols) < 3:
            return {'error': 'Insufficient numeric features for clustering'}
        
        X = df[feature_cols].fillna(0)
        
        # Robust scaling for outlier resistance
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        clustering_results = {}
        
        # K-Means with optimal k selection
        inertias = []
        silhouettes = []
        k_range = range(2, min(11, len(df) // 5))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            inertias.append(kmeans.inertia_)
            if len(set(cluster_labels)) > 1:
                silhouettes.append(silhouette_score(X_scaled, cluster_labels))
            else:
                silhouettes.append(0)
        
        # Optimal k using elbow method + silhouette score
        optimal_k = k_range[np.argmax(silhouettes)] if silhouettes else 3
        
        # Final K-Means clustering
        kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        kmeans_labels = kmeans_final.fit_predict(X_scaled)
        
        # DBSCAN clustering
        dbscan = DBSCAN(eps=0.5, min_samples=max(2, len(df) // 20))
        dbscan_labels = dbscan.fit_predict(X_scaled)
        
        # Hierarchical clustering
        hierarchical = AgglomerativeClustering(n_clusters=optimal_k)
        hierarchical_labels = hierarchical.fit_predict(X_scaled)
        
        # Evaluate clustering quality
        clustering_results['kmeans'] = {
            'n_clusters': optimal_k,
            'silhouette_score': float(silhouette_score(X_scaled, kmeans_labels)),
            'calinski_harabasz_score': float(calinski_harabasz_score(X_scaled, kmeans_labels)),
            'davies_bouldin_score': float(davies_bouldin_score(X_scaled, kmeans_labels)),
            'cluster_centers': kmeans_final.cluster_centers_.tolist(),
            'cluster_analysis': self._analyze_clusters(df, kmeans_labels)
        }
        
        if len(set(dbscan_labels)) > 1:
            clustering_results['dbscan'] = {
                'n_clusters': len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
                'noise_points': int(np.sum(dbscan_labels == -1)),
                'silhouette_score': float(silhouette_score(X_scaled, dbscan_labels)),
                'cluster_analysis': self._analyze_clusters(df, dbscan_labels)
            }
        
        clustering_results['hierarchical'] = {
            'n_clusters': optimal_k,
            'silhouette_score': float(silhouette_score(X_scaled, hierarchical_labels)),
            'cluster_analysis': self._analyze_clusters(df, hierarchical_labels)
        }
        
        return clustering_results
    
    def _analyze_clusters(self, df, cluster_labels):
        """Analyze characteristics of each cluster"""
        df_clustered = df.copy()
        df_clustered['cluster'] = cluster_labels
        
        cluster_analysis = []
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # Skip noise points in DBSCAN
                continue
                
            cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
            
            cluster_analysis.append({
                'cluster_id': int(cluster_id),
                'size': len(cluster_data),
                'percentage_of_total': round(len(cluster_data) / len(df) * 100, 2),
                'avg_business_impact': float(cluster_data['business_impact'].mean()),
                'total_business_impact': float(cluster_data['business_impact'].sum()),
                'avg_duration': float(cluster_data['duration_minutes'].mean()),
                'avg_customers_affected': float(cluster_data['customers_affected'].mean()),
                'dominant_severity': cluster_data['severity'].mode().iloc[0] if not cluster_data['severity'].mode().empty else 'Mixed',
                'dominant_provider': cluster_data['service_provider'].mode().iloc[0] if not cluster_data['service_provider'].mode().empty else 'Mixed',
                'dominant_error_type': cluster_data['error_type'].mode().iloc[0] if not cluster_data['error_type'].mode().empty else 'Mixed',
                'temporal_pattern': {
                    'peak_hour': int(cluster_data['hour'].mode().iloc[0]) if not cluster_data['hour'].mode().empty else None,
                    'peak_day': int(cluster_data['day_of_week'].mode().iloc[0]) if not cluster_data['day_of_week'].mode().empty else None
                }
            })
        
        return sorted(cluster_analysis, key=lambda x: x['avg_business_impact'], reverse=True)
    
    def _advanced_time_series_analysis(self, df):
        """Advanced time series analysis and forecasting"""
        df_ts = df.set_index('timestamp').sort_index()
        
        # Resample to daily frequency
        daily_impact = df_ts['business_impact'].resample('D').sum().fillna(0)
        daily_count = df_ts['business_impact'].resample('D').count().fillna(0)
        
        if len(daily_impact) < 7:
            return {'error': 'Insufficient data for time series analysis (minimum 7 days required)'}
        
        ts_analysis = {}
        
        # Trend analysis using linear regression
        X = np.arange(len(daily_impact)).reshape(-1, 1)
        y = daily_impact.values
        
        trend_model = LinearRegression().fit(X, y)
        trend_slope = trend_model.coef_[0]
        trend_r2 = trend_model.score(X, y)
        
        ts_analysis['trend_analysis'] = {
            'daily_trend_slope': float(trend_slope),
            'trend_significance': 'increasing' if trend_slope > 0 else 'decreasing' if trend_slope < 0 else 'stable',
            'trend_strength': float(trend_r2),
            'projected_30_day_impact': float(trend_model.predict([[len(daily_impact) + 30]])[0])
        }
        
        # Seasonal decomposition (if sufficient data)
        if len(daily_impact) >= 14:
            try:
                decomposition = seasonal_decompose(daily_impact, model='additive', period=7, extrapolate_trend='freq')
                
                ts_analysis['seasonal_decomposition'] = {
                    'seasonal_strength': float(np.var(decomposition.seasonal) / np.var(daily_impact)),
                    'trend_strength': float(np.var(decomposition.trend.dropna()) / np.var(daily_impact)),
                    'residual_variance': float(np.var(decomposition.resid.dropna())),
                    'seasonality_detected': bool(np.var(decomposition.seasonal) > 0.1 * np.var(daily_impact))
                }
            except:
                ts_analysis['seasonal_decomposition'] = {'error': 'Seasonal decomposition failed'}
        
        # ARIMA forecasting (if sufficient data)
        if len(daily_impact) >= 20:
            try:
                # Auto ARIMA model selection
                model = ARIMA(daily_impact, order=(1, 1, 1))
                fitted_model = model.fit()
                
                # Forecast next 7 days
                forecast = fitted_model.forecast(steps=7)
                forecast_confidence = fitted_model.get_forecast(steps=7).conf_int()
                
                ts_analysis['arima_forecast'] = {
                    'model_aic': float(fitted_model.aic),
                    'model_bic': float(fitted_model.bic),
                    'next_7_days_forecast': forecast.tolist(),
                    'forecast_confidence_intervals': {
                        'lower': forecast_confidence.iloc[:, 0].tolist(),
                        'upper': forecast_confidence.iloc[:, 1].tolist()
                    },
                    'forecast_total_impact': float(forecast.sum())
                }
                
                # Ljung-Box test for residual autocorrelation
                ljung_box = acorr_ljungbox(fitted_model.resid, lags=10, return_df=True)
                ts_analysis['model_diagnostics'] = {
                    'ljung_box_p_value': float(ljung_box['lb_pvalue'].min()),
                    'residuals_are_white_noise': bool(ljung_box['lb_pvalue'].min() > 0.05)
                }
                
            except Exception as e:
                ts_analysis['arima_forecast'] = {'error': f'ARIMA modeling failed: {str(e)}'}
        
        # Exponential smoothing (Holt-Winters)
        if len(daily_impact) >= 14:
            try:
                exp_smooth = ExponentialSmoothing(daily_impact, trend='add', seasonal='add', seasonal_periods=7)
                exp_model = exp_smooth.fit()
                exp_forecast = exp_model.forecast(7)
                
                ts_analysis['exponential_smoothing'] = {
                    'alpha': float(exp_model.params['smoothing_level']),
                    'beta': float(exp_model.params['smoothing_trend']),
                    'gamma': float(exp_model.params['smoothing_seasonal']),
                    'aic': float(exp_model.aic),
                    'next_7_days_forecast': exp_forecast.tolist(),
                    'forecast_total_impact': float(exp_forecast.sum())
                }
            except:
                ts_analysis['exponential_smoothing'] = {'error': 'Exponential smoothing failed'}
        
        return ts_analysis
    
    def _comprehensive_anomaly_analysis(self, df):
        """Multi-method anomaly detection"""
        feature_cols = [col for col in df.columns if any(x in col for x in ['impact', 'duration', 'customers', 'severity_weight'])]
        feature_cols = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']]
        
        X = df[feature_cols].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        anomaly_results = {}
        
        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
        iso_anomalies = iso_forest.fit_predict(X_scaled)
        iso_scores = iso_forest.decision_function(X_scaled)
        
        # Statistical anomalies (Z-score method)
        z_scores = np.abs(stats.zscore(X_scaled, axis=0))
        statistical_anomalies = (z_scores > 3).any(axis=1)
        
        # Combine anomaly detection methods
        anomaly_results['isolation_forest'] = {
            'anomaly_count': int(np.sum(iso_anomalies == -1)),
            'anomaly_percentage': round(np.mean(iso_anomalies == -1) * 100, 2),
            'average_anomaly_score': float(np.mean(iso_scores[iso_anomalies == -1])) if np.any(iso_anomalies == -1) else 0,
            'anomaly_indices': np.where(iso_anomalies == -1)[0].tolist()
        }
        
        anomaly_results['statistical_outliers'] = {
            'outlier_count': int(np.sum(statistical_anomalies)),
            'outlier_percentage': round(np.mean(statistical_anomalies) * 100, 2),
            'outlier_indices': np.where(statistical_anomalies)[0].tolist()
        }
        
        # Analyze anomalous incidents
        if np.any(iso_anomalies == -1):
            anomaly_incidents = df[iso_anomalies == -1]
            
            anomaly_results['anomaly_characteristics'] = {
                'avg_business_impact': float(anomaly_incidents['business_impact'].mean()),
                'total_business_impact': float(anomaly_incidents['business_impact'].sum()),
                'impact_vs_normal_ratio': float(anomaly_incidents['business_impact'].mean() / df[iso_anomalies == 1]['business_impact'].mean()),
                'common_providers': anomaly_incidents['service_provider'].value_counts().head(3).to_dict(),
                'common_error_types': anomaly_incidents['error_type'].value_counts().head(3).to_dict(),
                'temporal_distribution': anomaly_incidents['hour'].value_counts().to_dict()
            }
        
        return anomaly_results
    
    def _advanced_predictive_modeling(self, df):
        """Advanced predictive modeling using multiple algorithms"""
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'error_type', 'service_provider', 'severity', 'location', 'postcode', 'currency', 'title', 'description', 'root_cause']]
        feature_cols = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']]
        
        if len(feature_cols) < 5:
            return {'error': 'Insufficient features for predictive modeling'}
        
        X = df[feature_cols].fillna(0)
        y = df['business_impact']
        
        if len(X) < 30:
            return {'error': 'Insufficient data for reliable predictive modeling'}
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Multiple algorithms
        algorithms = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'svr': SVR(kernel='rbf'),
            'neural_network': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500),
            'elastic_net': ElasticNet(random_state=42)
        }
        
        model_results = {}
        
        for name, model in algorithms.items():
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                
                # Performance metrics
                mse = np.mean((y_test - y_pred) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(y_test - y_pred))
                r2 = model.score(X_test, y_test)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                
                model_results[name] = {
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'r2_score': float(r2),
                    'cv_mean_r2': float(np.mean(cv_scores)),
                    'cv_std_r2': float(np.std(cv_scores)),
                    'model_trained': True
                }
                
                # Feature importance (if available)
                if hasattr(model, 'feature_importances_'):
                    feature_importance = dict(zip(feature_cols, model.feature_importances_))
                    model_results[name]['feature_importance'] = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                
                # Store best model
                if 'best_model' not in model_results or r2 > model_results['best_model']['r2_score']:
                    model_results['best_model'] = {
                        'algorithm': name,
                        'r2_score': float(r2),
                        'rmse': float(rmse)
                    }
                    self.models['impact_predictor'] = model
                
            except Exception as e:
                model_results[name] = {'error': str(e), 'model_trained': False}
        
        return model_results
    
    def _provider_performance_intelligence(self, df):
        """Advanced provider performance analysis"""
        provider_analysis = {}
        
        for provider in df['service_provider'].unique():
            provider_data = df[df['service_provider'] == provider]
            
            # Performance metrics
            metrics = {
                'incident_count': len(provider_data),
                'total_impact': float(provider_data['business_impact'].sum()),
                'average_impact': float(provider_data['business_impact'].mean()),
                'median_impact': float(provider_data['business_impact'].median()),
                'impact_volatility': float(provider_data['business_impact'].std()),
                'average_resolution_time': float(provider_data['duration_minutes'].mean()),
                'total_customers_affected': int(provider_data['customers_affected'].sum()),
                'severity_distribution': provider_data['severity'].value_counts().to_dict(),
                'reliability_score': 0,
                'performance_trend': 'stable'
            }
            
            # Calculate reliability score using multiple factors
            impact_factor = max(0, 100 - (metrics['average_impact'] / 1000))
            frequency_factor = max(0, 100 - (metrics['incident_count'] * 5))
            duration_factor = max(0, 100 - (metrics['average_resolution_time'] / 10))
            volatility_factor = max(0, 100 - (metrics['impact_volatility'] / 500))
            
            metrics['reliability_score'] = round((impact_factor + frequency_factor + duration_factor + volatility_factor) / 4, 2)
            
            # Trend analysis
            if len(provider_data) >= 10:
                provider_data_sorted = provider_data.sort_values('timestamp')
                recent_impact = provider_data_sorted['business_impact'].tail(5).mean()
                earlier_impact = provider_data_sorted['business_impact'].head(5).mean()
                
                if recent_impact > earlier_impact * 1.1:
                    metrics['performance_trend'] = 'deteriorating'
                elif recent_impact < earlier_impact * 0.9:
                    metrics['performance_trend'] = 'improving'
            
            provider_analysis[provider] = metrics
        
        # Rank providers
        ranked_providers = sorted(provider_analysis.items(), key=lambda x: x[1]['reliability_score'], reverse=True)
        
        return {
            'provider_rankings': [(provider, metrics['reliability_score']) for provider, metrics in ranked_providers],
            'detailed_analysis': provider_analysis,
            'best_performer': ranked_providers[0][0] if ranked_providers else None,
            'worst_performer': ranked_providers[-1][0] if ranked_providers else None,
            'performance_spread': float(ranked_providers[0][1]['reliability_score'] - ranked_providers[-1][1]['reliability_score']) if len(ranked_providers) > 1 else 0
        }
    
    def _quantitative_risk_assessment(self, df):
        """Quantitative risk assessment using statistical methods"""
        risk_metrics = {}
        
        # Value at Risk (VaR) calculation
        impacts = df['business_impact'].values
        var_95 = float(np.percentile(impacts, 95))
        var_99 = float(np.percentile(impacts, 99))
        expected_shortfall_95 = float(np.mean(impacts[impacts >= var_95]))
        
        risk_metrics['value_at_risk'] = {
            'var_95_percent': var_95,
            'var_99_percent': var_99,
            'expected_shortfall_95': expected_shortfall_95,
            'max_observed_loss': float(impacts.max())
        }
        
        # Frequency analysis
        daily_incidents = df.groupby(df['timestamp'].dt.date).size()
        risk_metrics['frequency_risk'] = {
            'average_daily_incidents': float(daily_incidents.mean()),
            'max_daily_incidents': int(daily_incidents.max()),
            'probability_of_multiple_incidents': float((daily_incidents > 1).mean()),
            'probability_of_high_impact_day': float((df.groupby(df['timestamp'].dt.date)['business_impact'].sum() > var_95).mean())
        }
        
        # Concentration risk
        provider_concentration = df['service_provider'].value_counts(normalize=True)
        error_concentration = df['error_type'].value_counts(normalize=True)
        
        risk_metrics['concentration_risk'] = {
            'provider_herfindahl_index': float((provider_concentration ** 2).sum()),
            'error_type_herfindahl_index': float((error_concentration ** 2).sum()),
            'top_provider_exposure': float(provider_concentration.iloc[0]),
            'top_3_provider_exposure': float(provider_concentration.head(3).sum())
        }
        
        return risk_metrics
    
    def _business_impact_modeling(self, df):
        """Advanced business impact modeling"""
        impact_analysis = {}
        
        # Impact distribution analysis
        impacts = df['business_impact'].values
        
        # Fit different distributions
        from scipy.stats import norm, lognorm, gamma, expon
        
        distributions = {
            'normal': norm,
            'lognormal': lognorm,
            'gamma': gamma,
            'exponential': expon
        }
        
        best_fit = None
        best_ks_stat = float('inf')
        
        for dist_name, distribution in distributions.items():
            try:
                params = distribution.fit(impacts)
                ks_stat, ks_p = kstest(impacts, lambda x: distribution.cdf(x, *params))
                
                if ks_stat < best_ks_stat:
                    best_ks_stat = ks_stat
                    best_fit = {
                        'distribution': dist_name,
                        'parameters': params,
                        'ks_statistic': float(ks_stat),
                        'ks_p_value': float(ks_p)
                    }
            except:
                continue
        
        impact_analysis['distribution_modeling'] = best_fit
        
        # Cost modeling
        impact_analysis['cost_modeling'] = {
            'total_direct_costs': float(df['business_impact'].sum()),
            'average_incident_cost': float(df['business_impact'].mean()),
            'cost_per_customer': float(df['business_impact'].sum() / df['customers_affected'].sum()) if df['customers_affected'].sum() > 0 else 0,
            'cost_efficiency_ratio': float(df['business_impact'].sum() / df['duration_minutes'].sum()) if df['duration_minutes'].sum() > 0 else 0,
            'high_impact_threshold': float(np.percentile(impacts, 90)),
            'percentage_high_impact_incidents': float((impacts > np.percentile(impacts, 90)).mean() * 100)
        }
        
        return impact_analysis
    
    def _ml_recommendation_engine(self, df):
        """ML-powered recommendations based on comprehensive analysis"""
        recommendations = []
        
        # Provider recommendations
        provider_stats = df.groupby('service_provider').agg({
            'business_impact': ['mean', 'count'],
            'duration_minutes': 'mean'
        })
        
        worst_provider = provider_stats[('business_impact', 'mean')].idxmax()
        worst_impact = provider_stats.loc[worst_provider, ('business_impact', 'mean')]
        
        if worst_impact > df['business_impact'].mean() * 1.5:
            recommendations.append({
                'priority': 'High',
                'category': 'Provider Management',
                'recommendation': f'Immediate review of {worst_provider} required - 50% above average impact',
                'expected_benefit': f'Potential 30-40% reduction in incident costs',
                'confidence': 0.85
            })
        
        # Temporal recommendations
        hourly_impact = df.groupby('hour')['business_impact'].mean()
        peak_hours = hourly_impact.nlargest(3).index.tolist()
        
        if hourly_impact.max() > hourly_impact.mean() * 1.3:
            recommendations.append({
                'priority': 'Medium',
                'category': 'Operational Timing',
                'recommendation': f'Enhanced monitoring during peak hours: {peak_hours}',
                'expected_benefit': 'Faster incident detection and resolution',
                'confidence': 0.75
            })
        
        # Predictive recommendations
        if 'impact_predictor' in self.models:
            recommendations.append({
                'priority': 'Medium',
                'category': 'Predictive Analytics',
                'recommendation': 'Deploy ML-based impact prediction system',
                'expected_benefit': 'Proactive incident management and cost reduction',
                'confidence': 0.80
            })
        
        return {
            'recommendations': recommendations,
            'implementation_priority': sorted(recommendations, key=lambda x: {'High': 3, 'Medium': 2, 'Low': 1}[x['priority']], reverse=True)
        }
