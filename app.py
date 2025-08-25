#!/usr/bin/env python3

import os
import threading
import sqlite3
import time
import json
import logging
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import psycopg2.pool

# Disable Flask request logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

load_dotenv()

from mcp.incident_generator import IncidentGenerator
from mcp.trend_analyzer import TrendAnalyzer
from mcp.opportunity_discoverer import OpportunityDiscoverer

app = Flask(__name__)

def convert_numpy_types(obj):
    """Simple conversion that won't crash"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        try:
            if hasattr(obj, 'item'):
                return obj.item()
            elif str(type(obj)).startswith('<class \'numpy.'):
                return str(obj)
            else:
                return obj
        except:
            return str(obj)

class DatabaseManager:
    def __init__(self):
        self.db_url = os.getenv('DATABASE_URL')
        if not self.db_url:
            raise ValueError("DATABASE_URL environment variable is required")
        self.connection_pool = psycopg2.pool.ThreadedConnectionPool(1, 20, self.db_url)
        self.init_database()
        print("Connected to PostgreSQL database")
    
    def get_connection(self):
        return self.connection_pool.getconn()
    
    def return_connection(self, conn):
        self.connection_pool.putconn(conn)
    
    def init_database(self):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                # Create incidents table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS incidents (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        error_type VARCHAR(255),
                        severity VARCHAR(50),
                        service_provider VARCHAR(255),
                        business_impact INTEGER DEFAULT 0,
                        duration_minutes INTEGER DEFAULT 0,
                        customers_affected INTEGER DEFAULT 0,
                        resolution_status VARCHAR(50) DEFAULT 'new',
                        error_details TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create opportunities table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS opportunities (
                        id SERIAL PRIMARY KEY,
                        opportunity_type VARCHAR(255),
                        description TEXT,
                        revenue_potential BIGINT DEFAULT 0,
                        implementation_cost BIGINT DEFAULT 0,
                        roi_percentage INTEGER DEFAULT 0,
                        confidence_score INTEGER DEFAULT 0,
                        market_size VARCHAR(100),
                        analysis_run INTEGER DEFAULT 1,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create system_stats table - FIXED TO PREVENT DUPLICATES
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS system_stats (
                        id INTEGER PRIMARY KEY DEFAULT 1,
                        total_incidents INTEGER DEFAULT 0,
                        incidents_per_hour FLOAT DEFAULT 0,
                        last_analysis_at INTEGER DEFAULT 0,
                        analysis_run_count INTEGER DEFAULT 0,
                        system_start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        generation_stopped BOOLEAN DEFAULT FALSE,
                        ml_analysis_completed BOOLEAN DEFAULT FALSE,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # FIXED: Use INSERT ON CONFLICT to prevent duplicate stats records
                cur.execute("""
                    INSERT INTO system_stats (id, total_incidents, system_start_time)
                    VALUES (1, 0, CURRENT_TIMESTAMP)
                    ON CONFLICT (id) DO NOTHING
                """)
                
                conn.commit()
                print("Database tables created/verified")
        except Exception as e:
            print(f"Error initializing database: {e}")
            conn.rollback()
        finally:
            self.return_connection(conn)
    
    def save_incident(self, incident):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                # Insert new incident
                cur.execute("""
                    INSERT INTO incidents (
                        error_type, severity, service_provider, business_impact,
                        duration_minutes, customers_affected, resolution_status, error_details
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    incident.get('error_type'),
                    incident.get('severity'),
                    incident.get('service_provider'),
                    incident.get('business_impact', 0),
                    incident.get('duration_minutes', 0),
                    incident.get('customers_affected', 0),
                    incident.get('resolution_status', 'new'),
                    incident.get('error_details', '')
                ))
                
                incident_id = cur.fetchone()[0]
            
                # FIXED: Update the single system_stats record
                cur.execute("""
                    UPDATE system_stats 
                    SET total_incidents = (SELECT COUNT(*) FROM incidents),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = 1
                """)
                
                conn.commit()
            return incident_id
                
        except Exception as e:
            print(f"Error saving incident: {e}")
            conn.rollback()
            return None
        finally:
            self.return_connection(conn)
    
    def get_total_incidents(self):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                # FIXED: Always get actual count from incidents table
                cur.execute("SELECT COUNT(*) FROM incidents")
                actual_count = cur.fetchone()[0]
            
                # Update system_stats to match reality
                cur.execute("""
                    UPDATE system_stats 
                    SET total_incidents = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE id = 1
                """, (actual_count,))
                conn.commit()
                
                return actual_count
        except Exception as e:
            print(f"Error getting incident count: {e}")
            return 0
        finally:
            self.return_connection(conn)
    
    def get_recent_incidents(self, limit=100):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, timestamp, error_type, severity, service_provider, 
                           business_impact, duration_minutes, customers_affected, 
                           resolution_status, error_details, created_at
                        FROM incidents 
                    ORDER BY created_at DESC 
                    LIMIT %s
                """, (limit,))
                
                incidents = []
                for row in cur.fetchall():
                    incidents.append({
                        'id': row[0],
                        'timestamp': str(row[1]) if row[1] else str(row[10]),
                        'error_type': row[2],
                        'severity': row[3],
                        'service_provider': row[4],
                        'business_impact': int(row[5]) if row[5] else 0,
                        'duration_minutes': int(row[6]) if row[6] else 0,
                        'customers_affected': int(row[7]) if row[7] else 0,
                        'resolution_status': row[8],
                        'error_details': row[9],
                        'created_at': str(row[10])
                    })
                
                return incidents
        except Exception as e:
            print(f"Error fetching incidents: {e}")
            return []
        finally:
            self.return_connection(conn)
    
    def is_ml_analysis_completed(self):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT ml_analysis_completed FROM system_stats WHERE id = 1")
                result = cur.fetchone()
                return result[0] if result else False
        except:
            return False
        finally:
            self.return_connection(conn)
    
    def mark_ml_analysis_completed(self):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE system_stats 
                    SET ml_analysis_completed = TRUE, updated_at = CURRENT_TIMESTAMP
                    WHERE id = 1
                """)
                conn.commit()
                print("ML Analysis marked as completed")
        except Exception as e:
            print(f"Error marking ML analysis complete: {e}")
            conn.rollback()
        finally:
            self.return_connection(conn)
    
    def get_all_incidents_for_analysis(self):
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM incidents ORDER BY created_at")
                return [dict(row) for row in cur.fetchall()]
        except:
            return []
        finally:
            self.return_connection(conn)
    
    def save_business_opportunities(self, opportunities, analysis_run):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                # Remove old opportunities for this analysis run
                cur.execute("DELETE FROM opportunities WHERE analysis_run = %s", (analysis_run,))
                
                # Add new opportunities
                for opp in opportunities:
                    cur.execute("""
                        INSERT INTO opportunities (
                            opportunity_type, description, revenue_potential,
                            implementation_cost, roi_percentage, confidence_score,
                            market_size, analysis_run
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        opp.get('opportunity_type'),
                        opp.get('description'),
                        opp.get('revenue_potential', 0),
                        opp.get('implementation_cost', 0),
                        opp.get('roi_percentage', 0),
                        opp.get('confidence_score', 0),
                        opp.get('market_size'),
                        analysis_run
                    ))
                
                # Update stats
                cur.execute("""
                    UPDATE system_stats 
                    SET analysis_run_count = GREATEST(analysis_run_count, %s),
                        last_analysis_at = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = 1
                """, (analysis_run, analysis_run))
                
                conn.commit()
        except Exception as e:
            print(f"Error saving opportunities: {e}")
            conn.rollback()
        finally:
            self.return_connection(conn)
    
    def get_latest_business_opportunities(self):
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM opportunities 
                    WHERE analysis_run = (SELECT MAX(analysis_run) FROM opportunities)
                    ORDER BY revenue_potential DESC
                """)
                return [dict(row) for row in cur.fetchall()]
        except:
            return []
        finally:
            self.return_connection(conn)
    
    def get_system_stats(self):
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM system_stats WHERE id = 1")
                result = cur.fetchone()
                return dict(result) if result else {}
        except:
            return {}
        finally:
            self.return_connection(conn)
    
    def update_incidents_per_hour(self, rate):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE system_stats 
                    SET incidents_per_hour = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE id = 1
                """, (rate,))
            conn.commit()
        except Exception as e:
            print(f"Error updating incidents per hour: {e}")
            conn.rollback()
        finally:
            self.return_connection(conn)

class UKRetailEngine:
    def __init__(self):
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY required in .env file")
        
        self.db = DatabaseManager()
        
        # Get actual count from database
        actual_count = self.db.get_total_incidents()
        db_stats = self.db.get_system_stats()
        
        self.stats = {
            'total_incidents': actual_count,
            'incidents_per_hour': db_stats.get('incidents_per_hour', 0),
            'total_revenue_potential': 0,
            'opportunity_count': 0,
            'uptime_seconds': 0,
            'analysis_run_count': db_stats.get('analysis_run_count', 0),
            'generation_stopped': db_stats.get('generation_stopped', False),
            'ml_analysis_completed': db_stats.get('ml_analysis_completed', False),
            'max_incidents': 2500
        }
        
        self.generator = IncidentGenerator()
        self.analyzer = TrendAnalyzer()
        self.discoverer = OpportunityDiscoverer()
        
        try:
            self.start_time = datetime.fromisoformat(str(db_stats.get('system_start_time', datetime.now())))
        except:
            self.start_time = datetime.now()
        
        self.running = True
        self._load_existing_opportunities()
        
        # Log current status
        if actual_count >= 2500:
            print(f"SYSTEM INITIALIZED: {actual_count} incidents (CONTINUING POST-2500)")
            print(f"ML Analysis: {'COMPLETED' if self.stats['ml_analysis_completed'] else 'PENDING'}")
            print("Post-2500 mode: 1 incident every 15 minutes")
        else:
            print(f"SYSTEM INITIALIZED: {actual_count}/2500 incidents")
            print("ULTRA FAST generation: 1 second intervals")
            print(f"{2500 - actual_count} incidents remaining")
        
    def _load_existing_opportunities(self):
        opportunities = self.db.get_latest_business_opportunities()
        self.stats['opportunity_count'] = len(opportunities)
        self.stats['total_revenue_potential'] = sum(
            opp.get('revenue_potential', 0) for opp in opportunities
        )
        
    def start_engine(self):
        generation_thread = threading.Thread(target=self._incident_loop, daemon=True)
        generation_thread.start()
        
        analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        analysis_thread.start()
        
    def _incident_loop(self):
        while self.running:
            try:
                current_count = self.db.get_total_incidents()
                self.stats['total_incidents'] = current_count
                
                # After 2500 incidents, generate slowly (15 minutes = 900 seconds)
                if current_count >= 2500:
                    if not hasattr(self, '_slow_mode_logged'):
                        print("SLOW MODE: Generating 1 incident every 15 minutes after 2500")
                        self._slow_mode_logged = True
                    
                    print(f"Generating post-2500 incident #{current_count + 1}...")
                    incident = self.generator.generate_dynamic_incident()
                    incident_id = self.db.save_incident(incident)
                    
                    if incident_id:
                        print(f"Post-2500 incident #{current_count + 1} saved: {incident.get('error_type', 'Unknown')}")
                    
                    # Wait 15 minutes before next incident
                    time.sleep(900)
                    continue
                
                print(f"Generating incident #{current_count + 1}/2500...")
                incident = self.generator.generate_dynamic_incident()
                incident_id = self.db.save_incident(incident)
                
                if incident_id is None:
                    print("Generation failed")
                    time.sleep(10)
                    continue
                
                # Check if we hit exactly 2500 - run ML analysis ONCE
                if current_count + 1 == 2500 and not self.db.is_ml_analysis_completed():
                    print("TRIGGERING FINAL ML ANALYSIS FOR 2500 INCIDENTS")
                    self._discover_opportunities_final()
                    self.db.mark_ml_analysis_completed()
                
                # ULTRA FAST GENERATION until 2500: Only 1 second between incidents
                time.sleep(1)
                
            except Exception as e:
                print(f"INCIDENT GENERATION ERROR: {e}")
                time.sleep(10)
    
    def _analysis_loop(self):
        while self.running:
            try:
                uptime = (datetime.now() - self.start_time).total_seconds()
                self.stats['uptime_seconds'] = int(uptime)
                
                # Update actual count from database
                actual_count = self.db.get_total_incidents()
                self.stats['total_incidents'] = actual_count
                
                if uptime > 0:
                    incidents_per_hour = round((actual_count / uptime) * 3600, 2)
                    self.stats['incidents_per_hour'] = incidents_per_hour
                    self.db.update_incidents_per_hour(incidents_per_hour)
                
                time.sleep(30)
                
            except Exception as e:
                print(f"ANALYSIS ERROR: {e}")
                time.sleep(60)
    
    def _discover_opportunities_final(self):
        """Run ML analysis only once when reaching 2500 incidents"""
        try:
            all_incidents = self.db.get_all_incidents_for_analysis()
            
            print(f"RUNNING FINAL ML ANALYSIS on {len(all_incidents)} incidents...")
            
            opportunities = self.discoverer.discover_patterns(all_incidents)
            self.db.save_business_opportunities(opportunities, 1)  # Analysis run 1
            
            self.stats['opportunity_count'] = len(opportunities)
            self.stats['total_revenue_potential'] = sum(
                opp.get('revenue_potential', 0) for opp in opportunities
            )
            self.stats['analysis_run_count'] = 1
            
            print(f"FINAL ML ANALYSIS COMPLETE: {len(opportunities)} opportunities discovered")
            
        except Exception as e:
            print(f"FINAL ML ANALYSIS ERROR: {e}")

# The lines below were moved here to resolve the NameError
engine = UKRetailEngine()
engine.start_engine()

print(f"UK Retail Engine Status:")
print(f"   Current incidents: {engine.stats['total_incidents']}")
print(f"   Generation: {'POST-2500 MODE (15min intervals)' if engine.stats['total_incidents'] >= 2500 else 'ULTRA FAST (1s intervals)'}")
print(f"   ML Analysis: {'COMPLETED' if engine.stats['ml_analysis_completed'] else 'PENDING'}")
print(f"   Database: PostgreSQL Connected")
print("Starting Flask server...")

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/incidents/recent')
def get_recent_incidents():
    try:
        incidents = engine.db.get_recent_incidents()
        return jsonify(incidents)
    except Exception as e:
        print(f"API Error - incidents: {e}")
        return jsonify([])

@app.route('/api/incidents/full/<int:incident_id>')
def get_full_incident(incident_id):
    try:
        incidents = engine.db.get_recent_incidents()
        for incident in incidents:
            if incident.get('id') == incident_id:
                return jsonify(incident)
        return jsonify({'error': 'Incident not found'}), 404
    except Exception as e:
        print(f"API Error - full incident: {e}")
        return jsonify({'error': 'Server error'}), 500

@app.route('/api/trends/current')
def get_current_trends():
    try:
        incidents = engine.db.get_recent_incidents()
        if len(incidents) > 2:
            trends = engine.analyzer.analyze_patterns(incidents)
            trends = convert_numpy_types(trends)
            return jsonify(trends)
        return jsonify({'error': 'Insufficient data for trend analysis'})
    except Exception as e:
        print(f"API Error - trends: {e}")
        return jsonify({'error': 'Trend analysis failed', 'message': str(e)})

@app.route('/api/opportunities/latest')
def get_latest_opportunities():
    try:
        # Get all incidents
        all_incidents = engine.db.get_all_incidents_for_analysis()
        df = pd.DataFrame(all_incidents)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Prepare ML insights
        features_df = engine.discoverer._engineer_features(df)
        ml_models = engine.discoverer._train_ml_models(features_df)
        ml_insights = engine.discoverer._generate_ml_insights(features_df, ml_models)
        
        # Call the ML function and store output
        ml_opportunities = engine.discoverer._generate_ml_opportunities(df, ml_insights)
        
        # Save to database
        engine.db.save_business_opportunities(ml_opportunities, 1)
        
        # Convert and return as JSON
        opportunities = convert_numpy_types(ml_opportunities)
        return jsonify(opportunities)
        
    except Exception as e:
        print(f"API Error - opportunities: {e}")
        return jsonify([])

@app.route('/api/providers/reliability')
def get_provider_reliability():
    try:
        incidents = engine.db.get_recent_incidents()
        
        providers = {}
        for incident in incidents:
            provider = incident.get('service_provider', 'Unknown')
            if provider not in providers:
                providers[provider] = {'incidents': 0, 'total_impact': 0}
            providers[provider]['incidents'] += 1
            providers[provider]['total_impact'] += incident.get('business_impact', 0)
        
        reliability_scores = []
        for provider, data in providers.items():
            if data['incidents'] > 0:
                avg_impact = data['total_impact'] / data['incidents']
                reliability_score = max(0, 100 - (avg_impact / 1000))
                reliability_scores.append({
                    'provider': provider,
                    'incidents': data['incidents'],
                    'avg_impact': round(avg_impact, 2),
                    'reliability_score': round(reliability_score, 2)
                })
        
        result = sorted(reliability_scores, key=lambda x: x['reliability_score'], reverse=True)
        return jsonify(result)
    except Exception as e:
        print(f"API Error - providers: {e}")
        return jsonify([])

@app.route('/api/stats')
def get_stats():
    try:
        # Get fresh stats from database
        actual_count = engine.db.get_total_incidents()
        engine.stats['total_incidents'] = actual_count
        
        stats = {
            'total_incidents': actual_count,
            'incidents_per_hour': engine.stats['incidents_per_hour'],
            'total_revenue_potential': engine.stats['total_revenue_potential'],
            'opportunity_count': engine.stats['opportunity_count'],
            'uptime_seconds': engine.stats['uptime_seconds'],
            'analysis_run_count': engine.stats['analysis_run_count'],
            'generation_stopped': engine.stats['generation_stopped'],
            'ml_analysis_completed': engine.stats['ml_analysis_completed'],
            'max_incidents': 2500
        }
        
        return jsonify(stats)
    except Exception as e:
        print(f"API Error - stats: {e}")
        return jsonify({'error': 'Stats unavailable'})

@app.route('/api/analysis/detailed')
def get_detailed_analysis():
    try:
        incidents = engine.db.get_recent_incidents()
        
        if len(incidents) < 2:
            return jsonify({'error': 'Insufficient data for detailed analysis'})
        
        current_issues = {}
        severity_counts = {}
        provider_issues = {}
        
        for incident in incidents:
            error_type = incident.get('error_type', 'Unknown')
            if error_type not in current_issues:
                current_issues[error_type] = {
                    'count': 0,
                    'total_impact': 0,
                    'avg_duration': 0,
                    'customers_affected': 0,
                    'latest_occurrence': incident.get('timestamp'),
                    'severity_distribution': {}
                }
            
            current_issues[error_type]['count'] += 1
            current_issues[error_type]['total_impact'] += incident.get('business_impact', 0)
            current_issues[error_type]['avg_duration'] += incident.get('duration_minutes', 0)
            current_issues[error_type]['customers_affected'] += incident.get('customers_affected', 0)
            
            severity = incident.get('severity', 'Unknown')
            if severity not in current_issues[error_type]['severity_distribution']:
                current_issues[error_type]['severity_distribution'][severity] = 0
            current_issues[error_type]['severity_distribution'][severity] += 1
            
            if severity not in severity_counts:
                severity_counts[severity] = 0
            severity_counts[severity] += 1
            
            provider = incident.get('service_provider', 'Unknown')
            if provider not in provider_issues:
                provider_issues[provider] = {'count': 0, 'impact': 0}
            provider_issues[provider]['count'] += 1
            provider_issues[provider]['impact'] += incident.get('business_impact', 0)
        
        for error_type in current_issues:
            count = current_issues[error_type]['count']
            current_issues[error_type]['avg_duration'] = round(current_issues[error_type]['avg_duration'] / count, 1)
            current_issues[error_type]['avg_impact'] = round(current_issues[error_type]['total_impact'] / count, 0)
        
        critical_issues = sorted(
            [(k, v) for k, v in current_issues.items()], 
            key=lambda x: (x[1]['total_impact'], x[1]['count']), 
            reverse=True
        )[:5]
        
        top_providers_by_issues = sorted(
            [(k, v) for k, v in provider_issues.items()], 
            key=lambda x: x[1]['count'], 
            reverse=True
        )[:5]
        
        result = {
            'analysis_timestamp': datetime.now().isoformat(),
            'analyzed_incidents': len(incidents),
            'total_incidents_in_db': engine.stats['total_incidents'],
            'critical_issues': [
                {
                    'error_type': issue[0],
                    'frequency': issue[1]['count'],
                    'total_impact': issue[1]['total_impact'],
                    'avg_impact': issue[1]['avg_impact'],
                    'avg_duration': issue[1]['avg_duration'],
                    'customers_affected': issue[1]['customers_affected'],
                    'severity_breakdown': issue[1]['severity_distribution'],
                    'latest_occurrence': issue[1]['latest_occurrence']
                } for issue in critical_issues
            ],
            'severity_distribution': severity_counts,
            'provider_performance': [
                {
                    'provider': provider[0],
                    'incident_count': provider[1]['count'],
                    'total_impact': provider[1]['impact'],
                    'avg_impact_per_incident': round(provider[1]['impact'] / provider[1]['count'], 0)
                } for provider in top_providers_by_issues
            ],
            'summary': {
                'most_frequent_issue': critical_issues[0][0] if critical_issues else 'No data',
                'highest_impact_issue': max(current_issues.items(), key=lambda x: x[1]['total_impact'])[0] if current_issues else 'No data',
                'most_problematic_provider': top_providers_by_issues[0][0] if top_providers_by_issues else 'No data',
                'total_business_impact': sum(incident.get('business_impact', 0) for incident in incidents),
                'total_customers_affected': sum(incident.get('customers_affected', 0) for incident in incidents)
            }
        }
        
        return jsonify(result)
    except Exception as e:
        print(f"API Error - detailed analysis: {e}")
        return jsonify({'error': 'Analysis failed'})

@app.route('/debug/database')
def debug_database():
    try:
        db = engine.db
        conn = db.get_connection()
        
        debug_info = {
            'database_url': 'Connected' if db.db_url else 'Not set',
            'connection_status': 'OK',
            'tables': {},
            'incidents': {},
            'system_stats': {},
            'opportunities': {}
        }
        
        with conn.cursor() as cur:
            # Check incidents table
            cur.execute("SELECT COUNT(*) FROM incidents")
            incidents_count = cur.fetchone()[0]
            
            cur.execute("SELECT * FROM incidents ORDER BY created_at DESC LIMIT 5")
            recent_incidents = cur.fetchall()
            
            debug_info['incidents'] = {
                'total_count': incidents_count,
                'recent_5': [list(row) for row in recent_incidents]
            }
            
            # Check system_stats table
            cur.execute("SELECT * FROM system_stats")
            stats_rows = cur.fetchall()
            
            debug_info['system_stats'] = {
                'count': len(stats_rows),
                'data': [list(row) for row in stats_rows]
            }
            
            # Check opportunities table
            cur.execute("SELECT COUNT(*) FROM opportunities")
            opp_count = cur.fetchone()[0]
            
            debug_info['opportunities'] = {
                'total_count': opp_count
            }
            
            # Check table structures
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            tables = cur.fetchall()
            debug_info['tables'] = [row[0] for row in tables]
        
        db.return_connection(conn)
        
        return f"""
        <html>
        <head><title>Database Debug</title></head>
        <body style="font-family: monospace; padding: 20px;">
        <h1>Database Debug Information</h1>
        <pre>{json.dumps(debug_info, indent=2, default=str)}</pre>
        <hr>
        <h2>Raw Engine Stats:</h2>
        <pre>{json.dumps(engine.stats, indent=2, default=str)}</pre>
        </body>
        </html>
        """
        
    except Exception as e:
        return f"""
        <html>
        <head><title>Database Debug Error</title></head>
        <body style="font-family: monospace;
padding: 20px;">
        <h1>Database Connection Error</h1>
        <p><strong>Error:</strong> {str(e)}</p>
        <p><strong>DATABASE_URL set:</strong> {'Yes' if os.getenv('DATABASE_URL') else 'No'}</p>
        <p><strong>OPENAI_API_KEY set:</strong> {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}</p>
        </body>
        </html>
        """

@app.route('/debug/test-connection')
def test_connection():
    try:
        import psycopg2
        db_url = os.getenv('DATABASE_URL')
        
        if not db_url:
            return "DATABASE_URL environment variable not set"
        
        conn = psycopg2.connect(db_url)
        with conn.cursor() as cur:
            cur.execute("SELECT version()")
            version = cur.fetchone()[0]
        conn.close()
        
        return f"✅ PostgreSQL Connection OK<br>Version: {version}"
        
    except Exception as e:
        return f"❌ Database Connection Failed<br>Error: {str(e)}"

@app.route('/api/health')
def health_check():
    try:
        actual_count = engine.db.get_total_incidents()
        result = {
            'status': 'healthy',
            'uptime': engine.stats['uptime_seconds'],
            'total_incidents': actual_count,
            'incidents_remaining': max(0, 2500 - actual_count) if actual_count < 2500 else 0,
            'progress_percentage': round((min(actual_count, 2500) / 2500) * 100, 1),
            'generation_mode': 'POST-2500 (15min intervals)' if actual_count >= 2500 else 'ULTRA FAST (1s intervals)',
            'ml_analysis_completed': engine.stats['ml_analysis_completed'],
            'database_connected': True,
            'openai_configured': bool(os.getenv('OPENAI_API_KEY')),
            'analysis_runs_completed': engine.stats['analysis_run_count'],
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
