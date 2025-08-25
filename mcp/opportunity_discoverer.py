#!/usr/bin/env python3

import pandas as pd
import numpy as np
from openai import OpenAI
import os
import json
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class OpportunityDiscoverer:
    def __init__(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = None
        
        # Initialize ML models
        self.impact_predictor = None
        self.severity_classifier = None
        self.incident_clusterer = None
        self.anomaly_detector = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def discover_patterns(self, incidents):
        """Discover business opportunities using advanced ML models"""
        if len(incidents) < 50:
            return []
        
        print(f"ML: Training models on {len(incidents)} incidents")
        
        df = pd.DataFrame(incidents)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Prepare features for ML
        features_df = self._engineer_features(df)
        
        # Train multiple ML models
        ml_models = self._train_ml_models(features_df)
        
        # Generate ML-driven insights
        ml_insights = self._generate_ml_insights(features_df, ml_models)
        
        # Use ML insights + OpenAI for opportunities
        opportunities = self._generate_ml_opportunities(df, ml_insights)
        
        return opportunities[:5] if len(opportunities) >= 3 else opportunities + [{'title': 'Additional Analysis Required', 'description': 'Insufficient data for additional opportunities'}]
    
    def _engineer_features(self, df):
        """Engineer features for ML models without location"""
        features_df = df.copy()
        
        # Temporal features
        features_df['hour'] = features_df['timestamp'].dt.hour
        features_df['day_of_week'] = features_df['timestamp'].dt.dayofweek
        features_df['month'] = features_df['timestamp'].dt.month
        features_df['is_weekend'] = features_df['day_of_week'].isin([5, 6]).astype(int)
        features_df['is_business_hours'] = features_df['hour'].between(9, 17).astype(int)
        
        # Encode categorical variables (excluding location)
        categorical_columns = ['service_provider', 'error_type', 'severity']
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                features_df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(features_df[col].astype(str))
            else:
                # Handle unseen categories
                known_categories = set(self.label_encoders[col].classes_)
                features_df[f'{col}_encoded'] = features_df[col].apply(
                    lambda x: self.label_encoders[col].transform([str(x)])[0] if str(x) in known_categories else -1
                )
        
        # Business impact ratios
        features_df['impact_per_customer'] = features_df['business_impact'] / np.maximum(features_df['customers_affected'], 1)
        features_df['impact_per_minute'] = features_df['business_impact'] / np.maximum(features_df['duration_minutes'], 1)
        
        # Provider performance metrics
        provider_stats = features_df.groupby('service_provider').agg({
            'business_impact': ['mean', 'std', 'count'],
            'duration_minutes': 'mean',
            'customers_affected': 'mean'
        }).fillna(0)
        
        provider_stats.columns = ['provider_avg_impact', 'provider_impact_std', 'provider_incident_count', 
                                'provider_avg_duration', 'provider_avg_customers']
        
        features_df = features_df.merge(provider_stats, left_on='service_provider', right_index=True, how='left')
        
        return features_df
    
    def _train_ml_models(self, features_df):
        """Train multiple ML models for pattern discovery without location"""
        models = {}
        
        # Prepare feature matrix (excluding location-related features)
        feature_columns = [col for col in features_df.columns if col.endswith('_encoded') or 
                          col in ['hour', 'day_of_week', 'month', 'is_weekend', 'is_business_hours',
                                'impact_per_customer', 'impact_per_minute', 'provider_avg_impact',
                                'provider_impact_std', 'provider_incident_count', 'provider_avg_duration',
                                'provider_avg_customers']]
        
        X = features_df[feature_columns].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # 1. Business Impact Prediction Model (XGBoost)
        y_impact = features_df['business_impact']
        if len(y_impact.unique()) > 1:
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_impact, test_size=0.2, random_state=42)
            
            models['impact_predictor'] = xgb.XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
            )
            models['impact_predictor'].fit(X_train, y_train)
            
            # Get feature importance for business insights
            feature_importance = dict(zip(feature_columns, models['impact_predictor'].feature_importances_))
            models['impact_features'] = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Model performance
            y_pred = models['impact_predictor'].predict(X_test)
            models['impact_rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # 2. Severity Classification Model (Random Forest)
        y_severity = features_df['severity_encoded']
        if len(y_severity.unique()) > 1:
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_severity, test_size=0.2, random_state=42)
            
            models['severity_classifier'] = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42
            )
            models['severity_classifier'].fit(X_train, y_train)
            
            # Classification performance
            y_pred = models['severity_classifier'].predict(X_test)
            models['severity_accuracy'] = (y_pred == y_test).mean()
        
        # 3. Incident Clustering (K-Means)
        optimal_clusters = min(8, len(features_df) // 10)  # Dynamic cluster count
        if optimal_clusters >= 2:
            models['clusterer'] = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
            cluster_labels = models['clusterer'].fit_predict(X_scaled)
            features_df['cluster'] = cluster_labels
            
            # Analyze cluster characteristics (without location)
            cluster_analysis = features_df.groupby('cluster').agg({
                'business_impact': ['mean', 'count'],
                'duration_minutes': 'mean',
                'customers_affected': 'mean',
                'service_provider': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown',
                'error_type': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
            }).round(2)
            
            models['cluster_analysis'] = cluster_analysis
        
        # 4. Advanced Anomaly Detection
        models['anomaly_detector'] = IsolationForest(
            contamination=0.1, random_state=42, n_estimators=100
        )
        anomaly_scores = models['anomaly_detector'].fit_predict(X_scaled)
        features_df['is_anomaly'] = (anomaly_scores == -1).astype(int)
        models['anomaly_incidents'] = features_df[features_df['is_anomaly'] == 1]
        
        # 5. Time Series Pattern Detection
        features_df_sorted = features_df.sort_values('timestamp')
        daily_impact = features_df_sorted.groupby(features_df_sorted['timestamp'].dt.date)['business_impact'].sum()
        
        if len(daily_impact) > 7:  # Need at least a week of data
            # Simple trend analysis
            X_days = np.arange(len(daily_impact)).reshape(-1, 1)
            y_impact_daily = daily_impact.values
            
            models['trend_predictor'] = LinearRegression()
            models['trend_predictor'].fit(X_days, y_impact_daily)
            
            # Predict next 30 days
            future_days = np.arange(len(daily_impact), len(daily_impact) + 30).reshape(-1, 1)
            future_predictions = models['trend_predictor'].predict(future_days)
            models['future_impact_prediction'] = future_predictions.sum()
        
        return models
    
    def _generate_ml_insights(self, features_df, models):
        """Generate insights from trained ML models without location"""
        insights = {}
        
        # Impact prediction insights
        if 'impact_predictor' in models:
            insights['most_important_factors'] = models['impact_features'][:5]
            insights['model_accuracy'] = f"RMSE: {models['impact_rmse']:,.0f}"
        
        # Clustering insights
        if 'cluster_analysis' in models:
            cluster_df = models['cluster_analysis']
            # Find the most expensive cluster
            most_expensive_cluster = cluster_df[('business_impact', 'mean')].idxmax()
            insights['highest_cost_cluster'] = {
                'cluster_id': most_expensive_cluster,
                'avg_impact': cluster_df.loc[most_expensive_cluster, ('business_impact', 'mean')],
                'incident_count': cluster_df.loc[most_expensive_cluster, ('business_impact', 'count')],
                'dominant_provider': cluster_df.loc[most_expensive_cluster, ('service_provider', '<lambda>')],
                'dominant_error': cluster_df.loc[most_expensive_cluster, ('error_type', '<lambda>')]
            }
        
        # Anomaly insights
        if 'anomaly_incidents' in models:
            anomaly_df = models['anomaly_incidents']
            insights['anomaly_patterns'] = {
                'count': len(anomaly_df),
                'total_impact': anomaly_df['business_impact'].sum(),
                'avg_impact': anomaly_df['business_impact'].mean(),
                'common_providers': anomaly_df['service_provider'].value_counts().head(3).to_dict(),
                'common_errors': anomaly_df['error_type'].value_counts().head(3).to_dict()
            }
        
        # Predictive insights
        if 'future_impact_prediction' in models:
            insights['predicted_30_day_impact'] = models['future_impact_prediction']
        
        # Overall statistics
        insights['total_incidents'] = len(features_df)
        insights['total_impact'] = features_df['business_impact'].sum()
        insights['avg_impact'] = features_df['business_impact'].mean()
        insights['unique_providers'] = features_df['service_provider'].nunique()
        insights['unique_errors'] = features_df['error_type'].nunique()
        
        return insights
    
    def _generate_ml_opportunities(self, df, ml_insights):
        """Generate opportunities using ML insights and OpenAI without location"""
        
        prompt = f"""Analyze this ML data and write 3-5 actionable business opportunities: write practical effect, like which factor ahs most percentage name it and say if it is postitive or negative and how it can be an opportunity
        dont use jargon, take the name of the product and explain how this is an opportunity and why should be pursued, NO JARGON. Simple ENglish

{json.dumps(ml_insights, indent=2, default=str)}

Return JSON array with 3-5 opportunities. Each description should be 50 words explaining what the data shows and what action to take:

[
  {{
    "title": "Title based on data finding",
    "description": "50 words: what data shows and action to take",
    "category": "Category",
    "revenue_potential": 500000,
    "timeline": "months",
    "implementation_cost": 100000,
    "roi_percentage": 400,
    "confidence": 0.85,
    "risk_level": "Low/Medium/High"
  }}
]"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """Analyze ML data and return 3-5 business opportunities as JSON. Be specific about what the data shows and what actions to take.  write practical effect, like which factor ahs most percentage name it and say if it is postitive or negative and how it can be an opportunity
        dont use jargon, take the name of the product and explain how this is an opportunity and why should be pursued, NO JARGON. Simple ENglish"""},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2200,
                temperature=0.2
            )
            
            print("OpenAI Response Received")
            ai_response = response.choices[0].message.content.strip()
            
            # Parse AI response
            try:
                json_start = ai_response.find('[')
                json_end = ai_response.rfind(']') + 1
                if json_start != -1 and json_end != -1:
                    json_content = ai_response[json_start:json_end]
                    
                    # Clean the JSON - remove any currency symbols and percentage signs that might remain
                    json_content = json_content.replace('£', '').replace('$', '').replace('%', '').replace(' annually', '')
                    
                    opportunities = json.loads(json_content)
                    print(f"Successfully parsed {len(opportunities)} opportunities from OpenAI")
                    return self._validate_ml_opportunities(opportunities, ml_insights)
                else:
                    print("No valid JSON array found in OpenAI response")
                    raise json.JSONDecodeError("No valid JSON found", ai_response, 0)
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed: {e}")
                print(f"OpenAI response excerpt: {ai_response[:200]}...")
                return self._generate_ml_fallback_opportunities(ml_insights)
                
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return self._generate_ml_fallback_opportunities(ml_insights)
    
    def _generate_ml_fallback_opportunities(self, ml_insights):
        """Generate simple fallback opportunities when OpenAI fails"""
        
        # If OpenAI fails, just call the same function again with a simpler prompt
        simple_prompt = f"""Based on this data: {json.dumps(ml_insights, default=str)[:500]}...
        
Return 3 business opportunities as JSON array:  write practical effect, like which factor ahs most percentage name it and say if it is postitive or negative and how it can be an opportunity
        dont use jargon, take the name of the product and explain how this is an opportunity and why should be pursued, NO JARGON. Simple ENglish
[{{"title": "Opportunity", "description": "Simple description", "revenue_potential": 100000, "implementation_cost": 20000, "roi_percentage": 400, "confidence": 0.8, "risk_level": "Medium", "timeline": "6 months", "category": "Process"}}]"""
        
        try:
            if self.client:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": simple_prompt}],
                    max_tokens=800,
                    temperature=0.3
                )
                
                ai_response = response.choices[0].message.content.strip()
                json_start = ai_response.find('[')
                json_end = ai_response.rfind(']') + 1
                
                if json_start != -1 and json_end != -1:
                    json_content = ai_response[json_start:json_end]
                    opportunities = json.loads(json_content)
                    return opportunities
        except:
            pass
        
        # Last resort - return minimal opportunity
        return [{
            'title': 'Data Analysis Complete',
            'description': f'Analysis of {ml_insights.get("total_incidents", 0)} incidents complete. Contact team for detailed recommendations.',
            'category': 'Analysis',
            'revenue_potential': 50000,
            'confidence': 0.5,
            'timeline': '1 month',
            'risk_level': 'Low',
            'implementation_cost': 10000,
            'roi_percentage': 400
        }]
    
    def _validate_ml_opportunities(self, opportunities, ml_insights):
        """Validate and enhance ML-generated opportunities"""
        validated = []
        
        for opp in opportunities:
            if isinstance(opp, dict) and 'title' in opp:
                validated_opp = {
                    'title': opp.get('title', 'ML-Driven Opportunity'),
                    'description': opp.get('description', 'Opportunity identified through machine learning analysis'),
                    'category': opp.get('category', 'ML Analytics'),
                    'revenue_potential': max(5000, int(opp.get('revenue_potential', 25000))),
                    'confidence': min(0.95, max(0.3, float(opp.get('confidence', 0.7)))),
                    'timeline': opp.get('timeline', '4-6 months'),
                    'risk_level': opp.get('risk_level', 'Medium'),
                    'implementation_cost': max(1000, int(opp.get('implementation_cost', 8000))),
                    'roi_percentage': max(25, int(opp.get('roi_percentage', 150))),
                    'calculation_basis': opp.get('calculation_basis', 'ML model-based calculation'),
                    'discovery_timestamp': datetime.now().isoformat(),
                    'based_on_incidents': ml_insights['total_incidents']
                }
                validated.append(validated_opp)
        
        return validated
