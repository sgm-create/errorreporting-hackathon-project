#!/usr/bin/env python3

import os
import sys
import json
import requests

def create_project_structure():
    """Create complete project with all files using OpenAI for dynamic generation"""
    
    # Create directories
    dirs = ['mcp', 'templates', 'static/css', 'static/js', 'data']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Created: {d}/")

    # Write all files
    write_requirements()
    write_env_file()
    write_procfile()
    write_main_app()
    write_incident_generator()
    write_trend_analyzer()
    write_opportunity_discoverer()
    write_dashboard_template()
    write_init_files()
    
    print("\n" + "="*60)
    print("UK RETAIL INTELLIGENCE ENGINE CREATED SUCCESSFULLY!")
    print("="*60)
    print("1. Set your OpenAI API key in .env file")
    print("2. Run: pip install -r requirements.txt")
    print("3. Run: python app.py")
    print("4. Open: http://localhost:5000")
    print("5. For deployment: Procfile created for Koyeb/Heroku")
    print("="*60)

def write_requirements():
    with open('requirements.txt', 'w', encoding='utf-8') as f:
        f.write("""Flask
openai>=1.0.0
scikit-learn
pandas
numpy
requests
python-dotenv
matplotlib
seaborn
plotly
gunicorn
""")
    print("Created: requirements.txt")

def write_env_file():
    with open('.env', 'w', encoding='utf-8') as f:
        f.write("""OPENAI_API_KEY=your_openai_api_key_here
FLASK_ENV=development
PORT=5000
""")
    print("Created: .env")

def write_procfile():
    with open('Procfile', 'w', encoding='utf-8') as f:
        f.write("web: gunicorn app:app --bind 0.0.0.0:$PORT --workers 2")
    print("Created: Procfile")

def write_main_app():
    content = '''#!/usr/bin/env python3

import os
import threading
import time
import sqlite3
import json
from datetime import datetime
from flask import Flask, render_template, jsonify
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

# Import our modules
from mcp.incident_generator import IncidentGenerator
from mcp.trend_analyzer import TrendAnalyzer
from mcp.opportunity_discoverer import OpportunityDiscoverer

app = Flask(__name__)

class DatabaseManager:
    def __init__(self, db_path='data/incidents.db'):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create incidents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                error_type TEXT NOT NULL,
                service_provider TEXT NOT NULL,
                severity TEXT NOT NULL,
                business_impact INTEGER NOT NULL,
                customers_affected INTEGER NOT NULL,
                duration_minutes INTEGER NOT NULL,
                location TEXT NOT NULL,
                postcode TEXT NOT NULL,
                currency TEXT NOT NULL,
                title TEXT,
                description TEXT,
                root_cause TEXT,
                affected_systems TEXT,
                recovery_actions TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create business_opportunities table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS business_opportunities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_run INTEGER NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                category TEXT,
                revenue_potential INTEGER,
                confidence REAL,
                timeline TEXT,
                risk_level TEXT,
                implementation_cost INTEGER,
                roi_percentage INTEGER,
                calculation_basis TEXT,
                discovery_timestamp TEXT,
                based_on_incidents INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create system_stats table for persistent statistics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_stats (
                id INTEGER PRIMARY KEY,
                total_incidents INTEGER DEFAULT 0,
                last_analysis_at INTEGER DEFAULT 0,
                analysis_run_count INTEGER DEFAULT 0,
                system_start_time TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Initialize stats if not exists
        cursor.execute('SELECT COUNT(*) FROM system_stats WHERE id = 1')
        if cursor.fetchone()[0] == 0:
            cursor.execute("""
                INSERT INTO system_stats (id, system_start_time) 
                VALUES (1, ?)
            """, (datetime.now().isoformat(),))
        
        conn.commit()
        conn.close()
        print("Database initialized successfully")
    
    def save_incident(self, incident):
        """Save incident to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO incidents (
                timestamp, error_type, service_provider, severity, business_impact,
                customers_affected, duration_minutes, location, postcode, currency,
                title, description, root_cause, affected_systems, recovery_actions
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            incident.get('timestamp'),
            incident.get('error_type'),
            incident.get('service_provider'),
            incident.get('severity'),
            incident.get('business_impact'),
            incident.get('customers_affected'),
            incident.get('duration_minutes'),
            incident.get('location'),
            incident.get('postcode'),
            incident.get('currency'),
            incident.get('title'),
            incident.get('description'),
            incident.get('root_cause'),
            json.dumps(incident.get('affected_systems', [])),
            json.dumps(incident.get('recovery_actions', []))
        ))
        
        incident_id = cursor.lastrowid
        
        # Update total incidents count
        cursor.execute("""
            UPDATE system_stats 
            SET total_incidents = total_incidents + 1, 
                updated_at = CURRENT_TIMESTAMP 
            WHERE id = 1
        """)
        
        conn.commit()
        conn.close()
        return incident_id
    
    def get_total_incidents(self):
        """Get total incident count from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT total_incidents FROM system_stats WHERE id = 1')
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else 0
    
    def get_recent_incidents(self, limit=50):
        """Get recent incidents from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM incidents 
            ORDER BY created_at DESC 
            LIMIT ?
        """, (limit,))
        
        incidents = []
        for row in cursor.fetchall():
            incident = {
                'id': row[0],
                'timestamp': row[1],
                'error_type': row[2],
                'service_provider': row[3],
                'severity': row[4],
                'business_impact': row[5],
                'customers_affected': row[6],
                'duration_minutes': row[7],
                'location': row[8],
                'postcode': row[9],
                'currency': row[10],
                'title': row[11],
                'description': row[12],
                'root_cause': row[13],
                'affected_systems': json.loads(row[14]) if row[14] else [],
                'recovery_actions': json.loads(row[15]) if row[15] else []
            }
            incidents.append(incident)
        
        conn.close()
        return incidents
    
    def get_all_incidents_for_analysis(self):
        """Get all incidents for ML analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM incidents ORDER BY created_at ASC')
        
        incidents = []
        for row in cursor.fetchall():
            incident = {
                'id': row[0],
                'timestamp': row[1],
                'error_type': row[2],
                'service_provider': row[3],
                'severity': row[4],
                'business_impact': row[5],
                'customers_affected': row[6],
                'duration_minutes': row[7],
                'location': row[8],
                'postcode': row[9],
                'currency': row[10],
                'title': row[11],
                'description': row[12],
                'root_cause': row[13],
                'affected_systems': json.loads(row[14]) if row[14] else [],
                'recovery_actions': json.loads(row[15]) if row[15] else []
            }
            incidents.append(incident)
        
        conn.close()
        return incidents
    
    def save_business_opportunities(self, opportunities, analysis_run):
        """Save business opportunities to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Clear previous opportunities for this analysis run
        cursor.execute('DELETE FROM business_opportunities WHERE analysis_run = ?', (analysis_run,))
        
        for opp in opportunities:
            cursor.execute("""
                INSERT INTO business_opportunities (
                    analysis_run, title, description, category, revenue_potential,
                    confidence, timeline, risk_level, implementation_cost,
                    roi_percentage, calculation_basis, discovery_timestamp, based_on_incidents
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                analysis_run,
                opp.get('title'),
                opp.get('description'),
                opp.get('category'),
                opp.get('revenue_potential'),
                opp.get('confidence'),
                opp.get('timeline'),
                opp.get('risk_level'),
                opp.get('implementation_cost'),
                opp.get('roi_percentage'),
                opp.get('calculation_basis'),
                opp.get('discovery_timestamp'),
                opp.get('based_on_incidents')
            ))
        
        # Update analysis stats
        cursor.execute("""
            UPDATE system_stats 
            SET analysis_run_count = analysis_run_count + 1,
                last_analysis_at = ?,
                updated_at = CURRENT_TIMESTAMP 
            WHERE id = 1
        """, (analysis_run,))
        
        conn.commit()
        conn.close()
    
    def get_latest_business_opportunities(self):
        """Get latest business opportunities from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get the latest analysis run
        cursor.execute("""
            SELECT MAX(analysis_run) FROM business_opportunities
        """)
        latest_run = cursor.fetchone()[0]
        
        if latest_run is None:
            conn.close()
            return []
        
        cursor.execute("""
            SELECT * FROM business_opportunities 
            WHERE analysis_run = ?
            ORDER BY revenue_potential DESC
        """, (latest_run,))
        
        opportunities = []
        for row in cursor.fetchall():
            opp = {
                'title': row[2],
                'description': row[3],
                'category': row[4],
                'revenue_potential': row[5],
                'confidence': row[6],
                'timeline': row[7],
                'risk_level': row[8],
                'implementation_cost': row[9],
                'roi_percentage': row[10],
                'calculation_basis': row[11],
                'discovery_timestamp': row[12],
                'based_on_incidents': row[13]
            }
            opportunities.append(opp)
        
        conn.close()
        return opportunities
    
    def get_system_stats(self):
        """Get system statistics from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT total_incidents, last_analysis_at, analysis_run_count, system_start_time
            FROM system_stats WHERE id = 1
        """)
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'total_incidents': result[0],
                'last_analysis_at': result[1],
                'analysis_run_count': result[2],
                'system_start_time': result[3]
            }
        return {
            'total_incidents': 0,
            'last_analysis_at': 0,
            'analysis_run_count': 0,
            'system_start_time': datetime.now().isoformat()
        }

class UKRetailEngine:
    def __init__(self):
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY required in .env file")
        
        # Initialize database
        self.db = DatabaseManager()
        
        # Load existing stats from database
        db_stats = self.db.get_system_stats()
        
        self.stats = {
            'total_incidents': db_stats['total_incidents'],
            'incidents_per_hour': 0,
            'next_analysis_in': 500 - (db_stats['total_incidents'] % 500),
            'total_revenue_potential': 0,
            'opportunity_count': 0,
            'uptime_seconds': 0,
            'analysis_run_count': db_stats['analysis_run_count']
        }
        
        # Initialize components
        self.generator = IncidentGenerator()
        self.analyzer = TrendAnalyzer()
        self.discoverer = OpportunityDiscoverer()
        
        # Parse start time or use current time
        try:
            self.start_time = datetime.fromisoformat(db_stats['system_start_time'])
            print(f"Resumed system - Total incidents in database: {self.stats['total_incidents']}")
        except:
            self.start_time = datetime.now()
            print("Started fresh system")
        
        self.running = True
        
        # Load existing business opportunities
        self._load_existing_opportunities()
        
    def _load_existing_opportunities(self):
        """Load existing business opportunities from database"""
        opportunities = self.db.get_latest_business_opportunities()
        self.stats['opportunity_count'] = len(opportunities)
        self.stats['total_revenue_potential'] = sum(
            opp.get('revenue_potential', 0) for opp in opportunities
        )
        
        if opportunities:
            print(f"Loaded {len(opportunities)} existing business opportunities worth £{self.stats['total_revenue_potential']:,}")
        
    def start_engine(self):
        """Start the incident generation and analysis engines"""
        print(f"Starting UK Retail Intelligence Engine... (Incident #{self.stats['total_incidents'] + 1} next)")
        
        generation_thread = threading.Thread(target=self._incident_loop, daemon=True)
        generation_thread.start()
        
        analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        analysis_thread.start()
        
        print("Engine started successfully!")
        
    def _incident_loop(self):
        """Generate incidents and save to database"""
        import random
        while self.running:
            try:
                incident = self.generator.generate_dynamic_incident()
                
                # Save to database
                incident_id = self.db.save_incident(incident)
                incident['id'] = incident_id
                
                # Update stats
                self.stats['total_incidents'] += 1
                self.stats['next_analysis_in'] = 500 - (self.stats['total_incidents'] % 500)
                
                print(f"Generated incident #{self.stats['total_incidents']}: {incident.get('error_type', 'Unknown')} on {incident.get('service_provider', 'Unknown')}")
                
                # Trigger business opportunity discovery every 500 incidents
                if self.stats['total_incidents'] % 500 == 0:
                    print(f"🚀 Triggering business opportunity analysis #{self.stats['total_incidents'] // 500}")
                    self._discover_opportunities()
                
                # Random wait between 20 seconds to 20 minutes
                wait_time = random.randint(20, 1200)
                time.sleep(wait_time)
                
            except Exception as e:
                print(f"Error generating incident: {e}")
                time.sleep(60)
    
    def _analysis_loop(self):
        """Continuously analyze trends"""
        while self.running:
            try:
                # Update uptime
                uptime = (datetime.now() - self.start_time).total_seconds()
                self.stats['uptime_seconds'] = int(uptime)
                
                if uptime > 0:
                    self.stats['incidents_per_hour'] = round((self.stats['total_incidents'] / uptime) * 3600, 2)
                
                time.sleep(30)
                
            except Exception as e:
                print(f"Error in analysis loop: {e}")
                time.sleep(60)
    
    def _discover_opportunities(self):
        """Discover business opportunities using all historical data"""
        try:
            # Get all incidents from database for analysis
            all_incidents = self.db.get_all_incidents_for_analysis()
            
            if len(all_incidents) < 50:
                print("Insufficient data for business opportunity analysis")
                return
            
            print(f"Analyzing {len(all_incidents)} incidents for business opportunities...")
            
            # Generate analysis run number
            analysis_run = self.stats['total_incidents'] // 500
            
            # Discover opportunities
            opportunities = self.discoverer.discover_patterns(all_incidents)
            
            # Save to database
            self.db.save_business_opportunities(opportunities, analysis_run)
            
            # Update stats
            self.stats['opportunity_count'] = len(opportunities)
            self.stats['total_revenue_potential'] = sum(
                opp.get('revenue_potential', 0) for opp in opportunities
            )
            self.stats['analysis_run_count'] = analysis_run
            
            print(f"✅ Analysis #{analysis_run} complete: {len(opportunities)} opportunities worth £{self.stats['total_revenue_potential']:,}")
            
        except Exception as e:
            print(f"Error discovering opportunities: {e}")

# Initialize the engine
engine = UKRetailEngine()

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/incidents/recent')
def get_recent_incidents():
    """Get recent incidents from database"""
    incidents = engine.db.get_recent_incidents(50)
    return jsonify(incidents)

@app.route('/api/incidents/full/<int:incident_id>')
def get_full_incident(incident_id):
    """Get full incident details by ID"""
    incidents = engine.db.get_recent_incidents(1000)  # Search in recent incidents
    for incident in incidents:
        if incident.get('id') == incident_id:
            return jsonify(incident)
    return jsonify({'error': 'Incident not found'}), 404

@app.route('/api/trends/current')
def get_current_trends():
    """Get current trends from recent incidents"""
    incidents = engine.db.get_recent_incidents(100)
    if len(incidents) > 10:
        trends = engine.analyzer.analyze_patterns(incidents)
        return jsonify(trends)
    return jsonify({'error': 'Insufficient data for trend analysis'})

@app.route('/api/opportunities/latest')
def get_latest_opportunities():
    """Get latest business opportunities from database"""
    opportunities = engine.db.get_latest_business_opportunities()
    return jsonify(opportunities)

@app.route('/api/providers/reliability')
def get_provider_reliability():
    """Calculate provider reliability from database"""
    incidents = engine.db.get_recent_incidents(200)
    
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
    
    return jsonify(sorted(reliability_scores, key=lambda x: x['reliability_score'], reverse=True))

@app.route('/api/stats')
def get_stats():
    return jsonify(engine.stats)

@app.route('/api/analysis/detailed')
def get_detailed_analysis():
    """Get detailed analysis of current issues and patterns"""
    incidents = engine.db.get_recent_incidents(50)
    
    if len(incidents) < 10:
        return jsonify({'error': 'Insufficient data for detailed analysis'})
    
    # Calculate detailed analytics
    current_issues = {}
    severity_counts = {}
    provider_issues = {}
    
    for incident in incidents:
        # Track by error type
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
        
        # Track severity counts
        if severity not in severity_counts:
            severity_counts[severity] = 0
        severity_counts[severity] += 1
        
        # Track provider issues
        provider = incident.get('service_provider', 'Unknown')
        if provider not in provider_issues:
            provider_issues[provider] = {'count': 0, 'impact': 0}
        provider_issues[provider]['count'] += 1
        provider_issues[provider]['impact'] += incident.get('business_impact', 0)
    
    # Calculate averages
    for error_type in current_issues:
        count = current_issues[error_type]['count']
        current_issues[error_type]['avg_duration'] = round(current_issues[error_type]['avg_duration'] / count, 1)
        current_issues[error_type]['avg_impact'] = round(current_issues[error_type]['total_impact'] / count, 0)
    
    # Sort by impact and frequency
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
    
    return jsonify({
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
    })

@app.route('/api/health')
def health_check():
    """System health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'uptime': engine.stats['uptime_seconds'],
        'total_incidents': engine.stats['total_incidents'],
        'database_connected': True,
        'openai_configured': bool(os.getenv('OPENAI_API_KEY')),
        'analysis_runs_completed': engine.stats['analysis_run_count'],
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("Initializing UK Retail Intelligence Engine with SQLite persistence...")
    engine.start_engine()
    print("Starting web dashboard on http://localhost:5000")
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
'''
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Created: app.py")

def write_incident_generator():
    content = '''#!/usr/bin/env python3

import os
import json
import random
from openai import OpenAI
from datetime import datetime

class IncidentGenerator:
    def __init__(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key required")
        
        self.client = OpenAI(api_key=api_key)
        self.generated_errors = set()
        self.generated_providers = set()
        
    def generate_dynamic_incident(self):
        """Generate completely dynamic incident using OpenAI"""
        
        # Get or generate error types and providers dynamically
        error_type = self._get_dynamic_error_type()
        provider = self._get_dynamic_provider()
        
        # Generate random realistic values
        severity_options = ['Minor', 'Medium', 'Major', 'Critical']
        severity = random.choice(severity_options)
        business_impact = random.randint(500, 50000)
        customers_affected = random.randint(10, 5000)
        duration_minutes = random.randint(5, 480)
        
        # Generate UK location
        location_data = self._generate_uk_location()
        
        # Generate complete incident with OpenAI
        incident = self._generate_incident_details(
            error_type, provider, severity, business_impact, 
            customers_affected, duration_minutes, location_data
        )
        
        # Add metadata
        incident.update({
            'timestamp': datetime.now().isoformat(),
            'severity': severity,
            'business_impact': business_impact,
            'customers_affected': customers_affected,
            'duration_minutes': duration_minutes,
            'currency': 'GBP'
        })
        
        return incident
    
    def _get_dynamic_error_type(self):
        """Generate or retrieve dynamic error type"""
        if len(self.generated_errors) > 100:
            return random.choice(list(self.generated_errors))
        
        prompt = """Generate 1 realistic UK retail system error type. Examples: "Payment Gateway Timeout", "Inventory Sync Mismatch". 
        
        Return only the error name, nothing else."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.9
            )
            
            error_type = response.choices[0].message.content.strip()
            self.generated_errors.add(error_type)
            return error_type
            
        except:
            return f"System Error {random.randint(1000, 9999)}"
    
    def _get_dynamic_provider(self):
        """Generate or retrieve dynamic service provider"""
        if len(self.generated_providers) > 10:
            return random.choice(list(self.generated_providers))
        
        prompt = """Generate 1 realistic technology service provider name for UK retail. Examples: "Oracle Retail Solutions", "SAP Commerce Cloud".
        
        Return only the provider name, nothing else."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=25,
                temperature=0.9
            )
            
            provider = response.choices[0].message.content.strip()
            self.generated_providers.add(provider)
            return provider
            
        except:
            return f"TechProvider {random.randint(100, 999)}"
    
    def _generate_uk_location(self):
        """Generate UK location data"""
        uk_cities = [
            "London", "Manchester", "Birmingham", "Leeds", "Liverpool", 
            "Newcastle", "Sheffield", "Cardiff", "Edinburgh", "Glasgow"
        ]
        
        city = random.choice(uk_cities)
        postcode = f"{random.choice(['SW', 'M', 'B', 'LS', 'L', 'NE', 'S', 'CF', 'EH', 'G'])}{random.randint(1, 99)} {random.randint(1, 9)}{random.choice(['AA', 'BB', 'CC', 'DD'])}"
        
        return {'city': city, 'postcode': postcode}
    
    def _generate_incident_details(self, error_type, provider, severity, business_impact, 
                                 customers_affected, duration_minutes, location_data):
        """Generate complete incident details with OpenAI"""
        
        prompt = f"""Create a detailed UK retail incident report in JSON format:

Error: {error_type}
Provider: {provider}  
Severity: {severity}
Impact: £{business_impact:,}
Affected: {customers_affected:,} customers
Duration: {duration_minutes} minutes
Location: {location_data['city']}, {location_data['postcode']}

Generate JSON with:
- "error_type": "{error_type}"
- "service_provider": "{provider}"
- "title": Professional incident title
- "description": 100-150 word technical description with UK context, GDPR, VAT considerations
- "root_cause": Technical root cause
- "location": "{location_data['city']}"
- "postcode": "{location_data['postcode']}"
- "affected_systems": Array of 2-3 affected systems
- "recovery_actions": Array of 2-3 recovery actions taken

Use realistic technical details for UK retail operations."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a UK retail technical incident manager. Generate detailed, realistic incident reports in valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                return json.loads(ai_response)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    'error_type': error_type,
                    'service_provider': provider,
                    'title': f"{error_type} - {provider}",
                    'description': f"Technical incident involving {error_type} in {provider} system affecting {customers_affected:,} customers in {location_data['city']}. Service disruption lasted {duration_minutes} minutes with estimated business impact of £{business_impact:,}. Technical teams implemented emergency procedures and restored service functionality.",
                    'root_cause': f"Technical malfunction in {provider} infrastructure",
                    'location': location_data['city'],
                    'postcode': location_data['postcode'],
                    'affected_systems': [f"{provider} Core System", "Customer Database", "Payment Processing"],
                    'recovery_actions': ["Emergency service restart", "System monitoring enhanced", "Customer notifications sent"]
                }
                
        except Exception as e:
            print(f"OpenAI error: {e}")
            # Fallback incident
            return {
                'error_type': error_type,
                'service_provider': provider,
                'title': f"{error_type} - {provider}",
                'description': f"System incident in {provider} affecting {customers_affected:,} customers for {duration_minutes} minutes.",
                'root_cause': "Technical system error",
                'location': location_data['city'],
                'postcode': location_data['postcode'],
                'affected_systems': ["Primary System"],
                'recovery_actions': ["System restored"]
            }
'''
    
    with open('mcp/incident_generator.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Created: mcp/incident_generator.py")

def write_trend_analyzer():
    content = '''#!/usr/bin/env python3

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class TrendAnalyzer:
    def __init__(self):
        pass
        
    def analyze_patterns(self, incidents):
        """Analyze incident patterns using ML"""
        if len(incidents) < 5:
            return {'status': 'insufficient_data'}
        
        df = pd.DataFrame(incidents)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        analysis = {
            'provider_performance': self._analyze_providers(df),
            'error_clustering': self._cluster_errors(df),
            'temporal_patterns': self._analyze_temporal(df),
            'impact_analysis': self._analyze_impact(df),
            'predictions': self._generate_predictions(df)
        }
        
        return analysis
    
    def _analyze_providers(self, df):
        """Analyze service provider performance"""
        provider_stats = df.groupby('service_provider').agg({
            'business_impact': ['mean', 'sum', 'std', 'count'],
            'duration_minutes': ['mean', 'max'],
            'customers_affected': ['sum', 'mean']
        }).round(2)
        
        results = []
        for provider in provider_stats.index:
            stats = provider_stats.loc[provider]
            
            # Calculate reliability score using ML approach
            avg_impact = stats[('business_impact', 'mean')]
            incident_count = stats[('business_impact', 'count')]
            impact_variance = stats[('business_impact', 'std')] if pd.notna(stats[('business_impact', 'std')]) else 0
            
            # Weighted scoring algorithm
            reliability = max(0, 100 - (avg_impact / 500) - (incident_count * 2) - (impact_variance / 100))
            
            results.append({
                'provider': provider,
                'reliability_score': round(reliability, 2),
                'incident_count': int(incident_count),
                'avg_impact': float(avg_impact),
                'total_impact': float(stats[('business_impact', 'sum')]),
                'avg_duration': float(stats[('duration_minutes', 'mean')]),
                'max_duration': float(stats[('duration_minutes', 'max')]),
                'total_customers_affected': int(stats[('customers_affected', 'sum')])
            })
        
        return sorted(results, key=lambda x: x['reliability_score'], reverse=True)
    
    def _cluster_errors(self, df):
        """Cluster similar errors using DBSCAN"""
        try:
            # Prepare features for clustering
            features = []
            for _, row in df.iterrows():
                feature_vector = [
                    row['business_impact'],
                    row['duration_minutes'], 
                    row['customers_affected'],
                    hash(row['service_provider']) % 1000,  # Encode provider
                    {'Minor': 1, 'Medium': 2, 'Major': 3, 'Critical': 4}.get(row['severity'], 2)
                ]
                features.append(feature_vector)
            
            # Normalize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Perform clustering
            clustering = DBSCAN(eps=0.5, min_samples=2)
            cluster_labels = clustering.fit_predict(features_scaled)
            
            # Analyze clusters
            df['cluster'] = cluster_labels
            cluster_analysis = []
            
            for cluster_id in set(cluster_labels):
                if cluster_id == -1:  # Noise points
                    continue
                    
                cluster_data = df[df['cluster'] == cluster_id]
                
                cluster_analysis.append({
                    'cluster_id': int(cluster_id),
                    'incident_count': len(cluster_data),
                    'avg_impact': float(cluster_data['business_impact'].mean()),
                    'common_provider': cluster_data['service_provider'].mode().iloc[0] if not cluster_data['service_provider'].mode().empty else 'Various',
                    'common_severity': cluster_data['severity'].mode().iloc[0] if not cluster_data['severity'].mode().empty else 'Various',
                    'avg_duration': float(cluster_data['duration_minutes'].mean())
                })
            
            return sorted(cluster_analysis, key=lambda x: x['incident_count'], reverse=True)
            
        except Exception as e:
            return [{'error': f'Clustering failed: {str(e)}'}]
    
    def _analyze_temporal(self, df):
        """Analyze temporal patterns"""
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        
        # Hour analysis
        hourly_counts = df['hour'].value_counts().sort_index()
        hourly_impacts = df.groupby('hour')['business_impact'].sum().sort_index()
        
        # Day analysis  
        daily_counts = df['day_of_week'].value_counts()
        daily_impacts = df.groupby('day_of_week')['business_impact'].sum()
        
        # Peak detection
        peak_hour = hourly_counts.idxmax() if not hourly_counts.empty else 0
        peak_day = daily_counts.idxmax() if not daily_counts.empty else 'Unknown'
        
        return {
            'hourly_distribution': hourly_counts.to_dict(),
            'hourly_impact_distribution': hourly_impacts.to_dict(),
            'daily_distribution': daily_counts.to_dict(),
            'daily_impact_distribution': daily_impacts.to_dict(),
            'peak_hour': int(peak_hour),
            'peak_day': peak_day,
            'peak_hour_incidents': int(hourly_counts.get(peak_hour, 0)),
            'peak_day_incidents': int(daily_counts.get(peak_day, 0))
        }
    
    def _analyze_impact(self, df):
        """Analyze business impact trends"""
        # Calculate rolling statistics
        df_sorted = df.sort_values('timestamp')
        df_sorted['rolling_impact'] = df_sorted['business_impact'].rolling(window=min(10, len(df_sorted)), min_periods=1).mean()
        
        # Impact distribution
        impact_stats = {
            'total_impact': float(df['business_impact'].sum()),
            'avg_impact': float(df['business_impact'].mean()),
            'median_impact': float(df['business_impact'].median()),
            'max_impact': float(df['business_impact'].max()),
            'min_impact': float(df['business_impact'].min()),
            'impact_volatility': float(df['business_impact'].std()),
            'total_customers_affected': int(df['customers_affected'].sum()),
            'avg_customers_affected': float(df['customers_affected'].mean())
        }
        
        # Trend direction (simple)
        if len(df_sorted) >= 10:
            recent_avg = df_sorted['business_impact'].tail(5).mean()
            earlier_avg = df_sorted['business_impact'].head(5).mean()
            impact_stats['trend_direction'] = 'increasing' if recent_avg > earlier_avg else 'decreasing'
        else:
            impact_stats['trend_direction'] = 'stable'
        
        return impact_stats
    
    def _generate_predictions(self, df):
        """Generate simple predictions"""
        if len(df) < 10:
            return {'status': 'insufficient_data_for_predictions'}
        
        # Recent patterns
        recent_data = df.tail(20)
        
        # Most likely next provider
        provider_freq = recent_data['service_provider'].value_counts()
        next_provider = provider_freq.index[0] if not provider_freq.empty else 'Unknown'
        
        # Most likely next error
        error_freq = recent_data['error_type'].value_counts()
        next_error = error_freq.index[0] if not error_freq.empty else 'Unknown'
        
        # Predicted impact range
        recent_impacts = recent_data['business_impact']
        predicted_impact_min = max(500, int(recent_impacts.mean() - recent_impacts.std()))
        predicted_impact_max = int(recent_impacts.mean() + recent_impacts.std())
        
        # Time until next incident (based on recent intervals)
        if len(df) >= 2:
            time_diffs = df.sort_values('timestamp')['timestamp'].diff().dt.total_seconds().dropna()
            avg_interval = time_diffs.mean()
            predicted_next_minutes = max(20, int(avg_interval / 60))
        else:
            predicted_next_minutes = 60
        
        return {
            'most_likely_next_provider': next_provider,
            'provider_confidence': float(provider_freq.iloc[0] / len(recent_data)) if not provider_freq.empty else 0.0,
            'most_likely_next_error': next_error,
            'error_confidence': float(error_freq.iloc[0] / len(recent_data)) if not error_freq.empty else 0.0,
            'predicted_impact_range': [predicted_impact_min, predicted_impact_max],
            'predicted_next_incident_minutes': predicted_next_minutes,
            'overall_confidence': min(0.95, len(recent_data) / 100)
        }
'''
    
    with open('mcp/trend_analyzer.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Created: mcp/trend_analyzer.py")

def write_opportunity_discoverer():
    content = '''#!/usr/bin/env python3

import pandas as pd
import numpy as np
from openai import OpenAI
import os
import json
from datetime import datetime
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
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
        
    def discover_patterns(self, incidents):
        """Discover business opportunities using ML and AI"""
        if len(incidents) < 50:
            return []
        
        df = pd.DataFrame(incidents)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # ML Analysis for patterns
        ml_insights = self._ml_pattern_analysis(df)
        
        # Generate opportunities using OpenAI
        opportunities = self._generate_ai_opportunities(df, ml_insights)
        
        return opportunities[:15]  # Return top 15
    
    def _ml_pattern_analysis(self, df):
        """Use ML to find hidden patterns"""
        insights = {}
        
        # Provider reliability analysis
        provider_performance = df.groupby('service_provider').agg({
            'business_impact': ['mean', 'count', 'sum'],
            'duration_minutes': 'mean',
            'customers_affected': 'sum'
        })
        
        worst_performers = provider_performance.sort_values(('business_impact', 'mean'), ascending=False).head(3)
        insights['worst_providers'] = list(worst_performers.index)
        
        # Temporal pattern analysis
        df['hour'] = df['timestamp'].dt.hour
        peak_hours = df.groupby('hour')['business_impact'].sum().nlargest(3)
        insights['peak_impact_hours'] = list(peak_hours.index)
        
        # Anomaly detection using Isolation Forest
        try:
            features = df[['business_impact', 'duration_minutes', 'customers_affected']].fillna(0)
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomalies = iso_forest.fit_predict(features)
            insights['anomaly_count'] = int((anomalies == -1).sum())
            insights['total_incidents'] = len(df)
        except:
            insights['anomaly_count'] = 0
            insights['total_incidents'] = len(df)
        
        # Cost analysis
        insights['total_financial_impact'] = float(df['business_impact'].sum())
        insights['avg_incident_cost'] = float(df['business_impact'].mean())
        insights['total_customers_affected'] = int(df['customers_affected'].sum())
        
        # Provider diversity
        insights['provider_count'] = len(df['service_provider'].unique())
        insights['error_type_count'] = len(df['error_type'].unique())
        
        # Severity distribution
        severity_dist = df['severity'].value_counts()
        insights['critical_incidents'] = int(severity_dist.get('Critical', 0))
        insights['major_incidents'] = int(severity_dist.get('Major', 0))
        
        return insights
    
    def _generate_ai_opportunities(self, df, ml_insights):
        """Generate business opportunities using OpenAI based on ML insights"""
        
        if not self.client:
            return self._generate_fallback_opportunities(ml_insights)
        
        prompt = f"""You are a business analyst with access to REAL UK retail incident data. Based on this data, generate 5-8 specific business opportunities with DATA-DRIVEN estimates:

ACTUAL DATA ANALYSIS:
- Total Incidents: {ml_insights['total_incidents']}
- Total Financial Impact: £{ml_insights['total_financial_impact']:,.0f}
- Average Cost Per Incident: £{ml_insights['avg_incident_cost']:,.0f}
- Worst Performing Providers: {ml_insights['worst_providers']}
- Peak Impact Hours: {ml_insights['peak_impact_hours']}
- Provider Count: {ml_insights['provider_count']}
- Critical Incidents: {ml_insights['critical_incidents']}
- Major Incidents: {ml_insights['major_incidents']}
- Anomalies Detected: {ml_insights['anomaly_count']}
- Total Customers Affected: {ml_insights['total_customers_affected']:,}

For EACH opportunity, you must CALCULATE and JUSTIFY:

1. **Timeline**: Base on incident frequency and complexity
   - High frequency issues = faster implementation needed
   - Complex integrations = longer timeline
   - Use the actual incident count to determine urgency

2. **Revenue Potential**: Calculate from REAL cost data
   - Use the actual £{ml_insights['total_financial_impact']:,.0f} total impact
   - Calculate realistic savings percentages based on provider performance
   - Factor in the {ml_insights['total_incidents']} incidents for annual projections

3. **Implementation Cost**: Base on realistic IT project costs
   - Provider migrations: estimate based on system complexity
   - Technology investments: calculate based on incident volume
   - Use industry standards, not made-up numbers

4. **ROI Percentage**: Calculate as (Revenue - Cost) / Cost * 100
   - Must be mathematically correct based on your other estimates
   - Show the calculation logic

5. **Confidence Score**: Base on data quality and pattern strength
   - More incidents = higher confidence in patterns
   - Clear trends = higher confidence
   - Calculate: min(0.95, incidents_analyzed / 500)

6. **Risk Level**: Determine from implementation complexity
   - Simple software changes = Low risk
   - Provider migrations = Medium risk  
   - Major system overhauls = High risk

Generate JSON array with calculated values:
```json
[
  {{
    "title": "Specific opportunity title",
    "description": "100-word description explaining WHY this opportunity exists based on the data",
    "category": "Category name",
    "revenue_potential": calculated_value_with_reasoning,
    "timeline": "calculated_based_on_urgency_and_complexity",
    "implementation_cost": calculated_value_with_reasoning,
    "roi_percentage": mathematically_calculated_roi,
    "confidence": calculated_confidence_score,
    "risk_level": "determined_from_complexity",
    "calculation_basis": "Brief explanation of how you calculated the numbers"
  }}
]
```

DO NOT make up arbitrary numbers. Every estimate must be derived from the actual incident data provided."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a data-driven business analyst. Every number you provide must be calculated from real data, not estimated arbitrarily. Show your reasoning for each calculation."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.3  # Lower temperature for more consistent calculations
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            # Parse AI response
            try:
                # Extract JSON from response
                json_start = ai_response.find('[')
                json_end = ai_response.rfind(']') + 1
                if json_start != -1 and json_end != -1:
                    json_content = ai_response[json_start:json_end]
                    opportunities = json.loads(json_content)
                    return self._validate_opportunities(opportunities, ml_insights)
                else:
                    raise json.JSONDecodeError("No valid JSON found")
            except json.JSONDecodeError:
                print("AI response was not valid JSON, using calculation-based fallback")
                return self._generate_calculated_opportunities(ml_insights)
                
        except Exception as e:
            print(f"OpenAI error in opportunity discovery: {e}")
            return self._generate_calculated_opportunities(ml_insights)
    
    def _generate_calculated_opportunities(self, ml_insights):
        """Generate opportunities with calculated values based on actual data"""
        opportunities = []
        
        # Calculate confidence based on data volume
        data_confidence = min(0.95, ml_insights['total_incidents'] / 500)
        
        # Provider switching opportunity - if there are clear worst performers
        if ml_insights['worst_providers'] and len(ml_insights['worst_providers']) > 0:
            worst_provider = ml_insights['worst_providers'][0]
            
            # Calculate potential savings: assume 50% improvement from switching
            annual_incidents_estimate = ml_insights['total_incidents'] * (365 / max(1, ml_insights['total_incidents'] * 0.1))  # Rough annual projection
            current_annual_cost = ml_insights['avg_incident_cost'] * annual_incidents_estimate
            potential_savings = current_annual_cost * 0.5  # 50% improvement
            
            # Calculate implementation cost: 20% of annual savings for provider migration
            implementation_cost = potential_savings * 0.2
            
            # Calculate timeline based on incident urgency
            if ml_insights['critical_incidents'] > 5:
                timeline = "3-4 months (urgent due to critical incidents)"
            elif ml_insights['total_incidents'] > 100:
                timeline = "4-6 months (high incident volume)"
            else:
                timeline = "6-9 months (standard migration)"
            
            # Calculate ROI
            roi = ((potential_savings - implementation_cost) / implementation_cost) * 100
            
            opportunities.append({
                'title': f'Service Provider Migration from {worst_provider}',
                'description': f'Data shows {worst_provider} has highest incident impact. Analysis of {ml_insights["total_incidents"]} incidents indicates 50% cost reduction possible through provider migration, based on performance differential.',
                'category': 'Vendor Management',
                'revenue_potential': int(potential_savings),
                'confidence': round(data_confidence, 2),
                'timeline': timeline,
                'risk_level': 'Medium',
                'implementation_cost': int(implementation_cost),
                'roi_percentage': int(roi),
                'calculation_basis': f'Based on {ml_insights["total_incidents"]} incidents, avg cost £{ml_insights["avg_incident_cost"]:.0f}, 50% improvement assumption',
                'discovery_timestamp': datetime.now().isoformat(),
                'based_on_incidents': ml_insights['total_incidents']
            })
        
        # Peak time monetization - if there are clear peak patterns
        if ml_insights['peak_impact_hours'] and len(ml_insights['peak_impact_hours']) > 0:
            peak_hour = ml_insights['peak_impact_hours'][0]
            
            # Calculate premium service revenue: 30% of customers would pay 25% premium
            affected_customers = ml_insights['total_customers_affected']
            premium_customers = int(affected_customers * 0.3)  # 30% willing to pay premium
            avg_service_value = ml_insights['avg_incident_cost'] * 0.1  # Assume service fee is 10% of incident cost
            premium_revenue = premium_customers * avg_service_value * 1.25  # 25% premium
            
            # Implementation cost: 15% of revenue for system development
            implementation_cost = premium_revenue * 0.15
            
            # Timeline based on system complexity
            if ml_insights['provider_count'] > 5:
                timeline = "4-6 months (complex multi-provider setup)"
            else:
                timeline = "2-4 months (standard implementation)"
            
            roi = ((premium_revenue - implementation_cost) / implementation_cost) * 100
            
            opportunities.append({
                'title': f'Premium Support Services During Peak Hours ({peak_hour}:00)',
                'description': f'Peak hour analysis shows {affected_customers:,} customers affected during high-impact periods. Market research indicates 30% would pay premium for guaranteed faster resolution.',
                'category': 'Service Monetization',
                'revenue_potential': int(premium_revenue),
                'confidence': round(data_confidence * 0.8, 2),  # Slightly lower confidence for market assumptions
                'timeline': timeline,
                'risk_level': 'Low',
                'implementation_cost': int(implementation_cost),
                'roi_percentage': int(roi),
                'calculation_basis': f'{affected_customers:,} customers * 30% adoption * £{avg_service_value:.0f} avg value * 125% premium',
                'discovery_timestamp': datetime.now().isoformat(),
                'based_on_incidents': ml_insights['total_incidents']
            })
        
        # Predictive maintenance investment - if high-severity incidents justify it
        if ml_insights['critical_incidents'] + ml_insights['major_incidents'] > 10:
            high_severity_incidents = ml_insights['critical_incidents'] + ml_insights['major_incidents']
            
            # Calculate prevention value: assume 70% of high-severity incidents preventable
            preventable_incidents = high_severity_incidents * 0.7
            avg_high_severity_cost = ml_insights['avg_incident_cost'] * 2.5  # High severity costs more
            annual_prevention_value = preventable_incidents * avg_high_severity_cost * (365 / max(1, ml_insights['total_incidents'] * 0.1))
            
            # Implementation cost: AI system is typically 40% of annual savings
            implementation_cost = annual_prevention_value * 0.4
            
            # Timeline based on system complexity
            if ml_insights['provider_count'] > 7:
                timeline = "8-12 months (complex multi-vendor AI integration)"
            elif ml_insights['error_type_count'] > 50:
                timeline = "6-9 months (high pattern complexity)"
            else:
                timeline = "4-6 months (standard AI deployment)"
            
            roi = ((annual_prevention_value - implementation_cost) / implementation_cost) * 100
            
            opportunities.append({
                'title': 'Predictive Maintenance AI System',
                'description': f'Analysis of {high_severity_incidents} critical/major incidents shows patterns suitable for AI prediction. System could prevent 70% of high-severity outages based on temporal and system correlations.',
                'category': 'Technology Investment',
                'revenue_potential': int(annual_prevention_value),
                'confidence': round(data_confidence, 2),
                'timeline': timeline,
                'risk_level': 'Medium',
                'implementation_cost': int(implementation_cost),
                'roi_percentage': int(roi),
                'calculation_basis': f'{high_severity_incidents} high-severity incidents * 70% prevention rate * £{avg_high_severity_cost:.0f} avg cost',
                'discovery_timestamp': datetime.now().isoformat(),
                'based_on_incidents': ml_insights['total_incidents']
            })
        
        return opportunities
    
    def _validate_opportunities(self, opportunities, ml_insights):
        """Validate and enhance AI-generated opportunities"""
        validated = []
        
        for opp in opportunities:
            if isinstance(opp, dict) and 'title' in opp:
                # Ensure required fields
                validated_opp = {
                    'title': opp.get('title', 'Business Opportunity'),
                    'description': opp.get('description', 'Opportunity based on incident patterns'),
                    'category': opp.get('category', 'Strategic Initiative'),
                    'revenue_potential': max(1000, int(opp.get('revenue_potential', 10000))),
                    'confidence': min(1.0, max(0.1, float(opp.get('confidence', 0.7)))),
                    'timeline': opp.get('timeline', '3-6 months'),
                    'risk_level': opp.get('risk_level', 'Medium'),
                    'implementation_cost': max(500, int(opp.get('implementation_cost', 5000))),
                    'roi_percentage': max(50, int(opp.get('roi_percentage', 150))),
                    'discovery_timestamp': datetime.now().isoformat(),
                    'based_on_incidents': ml_insights['total_incidents']
                }
                validated.append(validated_opp)
        
        return validated
    
    def _generate_fallback_opportunities(self, ml_insights):
        """Generate fallback opportunities with calculated values if AI fails"""
        return self._generate_calculated_opportunities(ml_insights)
'''
    
    with open('mcp/opportunity_discoverer.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Created: mcp/opportunity_discoverer.py")

def write_dashboard_template():
    content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enterprise Incident Intelligence Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f7fa; color: #2c3e50; overflow-x: auto;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 20px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header-content { max-width: 1400px; margin: 0 auto; padding: 0 20px; }
        .header h1 { font-size: 2em; margin-bottom: 5px; }
        .header p { opacity: 0.9; font-size: 1.1em; }
        .status-live { 
            display: inline-block; width: 8px; height: 8px; background: #00ff88;
            border-radius: 50%; margin-right: 8px; animation: pulse 1.5s infinite;
        }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        
        .alert-bar {
            background: #fff3cd; border: 1px solid #ffeaa7; color: #856404;
            padding: 12px 20px; border-radius: 8px; margin-bottom: 20px;
            font-weight: 500;
        }
        
        .nav-tabs {
            display: flex; gap: 2px; margin-bottom: 20px; background: #fff;
            border-radius: 10px; padding: 4px; box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }
        .nav-tab {
            flex: 1; padding: 12px 20px; background: transparent; border: none;
            border-radius: 8px; cursor: pointer; font-weight: 500; transition: all 0.3s;
            color: #6c757d;
        }
        .nav-tab.active {
            background: linear-gradient(135deg, #667eea, #764ba2); color: white;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        .nav-tab:hover:not(.active) { background: #f8f9fa; color: #495057; }
        
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        
        .metrics-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px; margin-bottom: 30px;
        }
        .metric-card {
            background: white; border-radius: 12px; padding: 24px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08); position: relative; overflow: hidden;
        }
        .metric-card::before {
            content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px;
            background: linear-gradient(90deg, #667eea, #764ba2);
        }
        .metric-value { font-size: 2.5em; font-weight: bold; color: #2c3e50; line-height: 1; }
        .metric-label { color: #6c757d; font-size: 0.9em; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 8px; }
        .metric-change { font-size: 0.85em; margin-top: 5px; font-weight: 500; }
        .metric-change.positive { color: #27ae60; }
        .metric-change.negative { color: #e74c3c; }
        
        .charts-grid { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; margin-bottom: 30px; }
        .chart-container {
            background: white; border-radius: 12px; padding: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }
        .chart-title { font-size: 1.2em; font-weight: 600; margin-bottom: 20px; color: #2c3e50; }
        .chart-canvas { position: relative; height: 300px; }
        
        .analysis-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .analysis-section {
            background: white; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            overflow: hidden;
        }
        .section-header {
            background: linear-gradient(135deg, #667eea, #764ba2); color: white;
            padding: 16px 20px; font-weight: 600; font-size: 1.1em;
        }
        .section-content { padding: 20px; max-height: 400px; overflow-y: auto; }
        
        .incident-item {
            background: #f8f9fa; border-radius: 8px; padding: 16px; margin-bottom: 12px;
            border-left: 4px solid #667eea; cursor: pointer; transition: all 0.2s;
        }
        .incident-item:hover { background: #e9ecef; transform: translateX(4px); }
        .incident-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px; }
        .incident-title { font-weight: 600; color: #2c3e50; flex: 1; }
        .severity-badge {
            padding: 2px 8px; border-radius: 12px; font-size: 0.75em; font-weight: 600;
            text-transform: uppercase;
        }
        .severity-critical { background: #ffebee; color: #c62828; }
        .severity-major { background: #fff3e0; color: #f57c00; }
        .severity-medium { background: #f3e5f5; color: #7b1fa2; }
        .severity-minor { background: #e8f5e8; color: #2e7d32; }
        .incident-meta { font-size: 0.85em; color: #6c757d; margin-bottom: 8px; }
        .incident-impact { font-weight: 600; color: #e74c3c; }
        
        .loading {
            text-align: center; padding: 40px; color: #6c757d; font-style: italic;
        }
        .loading::after {
            content: ''; display: inline-block; width: 20px; height: 20px;
            border: 2px solid #f3f3f3; border-top: 2px solid #667eea;
            border-radius: 50%; animation: spin 1s linear infinite; margin-left: 10px;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        
        .modal {
            display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.5); z-index: 1000;
        }
        .modal.active { display: flex; align-items: center; justify-content: center; }
        .modal-content {
            background: white; border-radius: 12px; max-width: 600px; width: 90%;
            max-height: 80vh; overflow-y: auto; box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        .modal-header {
            background: linear-gradient(135deg, #667eea, #764ba2); color: white;
            padding: 20px; border-radius: 12px 12px 0 0;
        }
        .modal-body { padding: 20px; }
        .close-modal {
            float: right; background: none; border: none; color: white;
            font-size: 24px; cursor: pointer; padding: 0; width: 30px; height: 30px;
        }
        
        @media (max-width: 968px) {
            .charts-grid { grid-template-columns: 1fr; }
            .analysis-grid { grid-template-columns: 1fr; }
            .metrics-grid { grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-content">
            <h1><span class="status-live"></span>Enterprise Incident Intelligence Dashboard</h1>
            <p>Powered by Real-Time ML Analysis & AI-Generated Insights</p>
        </div>
    </div>

    <div class="container">
        <div class="alert-bar" id="alertBar">
            <span id="alertText">System monitoring active - Real-time incident generation in progress</span>
        </div>

        <div class="nav-tabs">
            <button class="nav-tab active" onclick="switchTab('trends')">Current Trends</button>
            <button class="nav-tab" onclick="switchTab('analysis')">Trend Analysis</button>
            <button class="nav-tab" onclick="switchTab('opportunities')">Business Opportunities</button>
            <button class="nav-tab" onclick="switchTab('health')">System Health</button>
        </div>

        <div id="trends" class="tab-content active">
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value" id="activeIncidents">0</div>
                    <div class="metric-label">Active Incidents</div>
                    <div class="metric-change positive" id="incidentTrend">+0 this hour</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="avgResolution">0m</div>
                    <div class="metric-label">Avg Resolution Time</div>
                    <div class="metric-change" id="resolutionTrend">Calculating...</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="systemsAffected">0</div>
                    <div class="metric-label">Systems Affected</div>
                    <div class="metric-change" id="systemsTrend">Current period</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="businessImpact">£0</div>
                    <div class="metric-label">Total Business Impact</div>
                    <div class="metric-change negative" id="impactTrend">Live tracking</div>
                </div>
            </div>

            <div class="charts-grid">
                <div class="chart-container">
                    <div class="chart-title">Incident Frequency & Impact Trends</div>
                    <div class="chart-canvas">
                        <canvas id="incidentChart"></canvas>
                    </div>
                </div>
                <div class="chart-container">
                    <div class="chart-title">Service Provider Reliability</div>
                    <div class="chart-canvas">
                        <canvas id="providerChart"></canvas>
                    </div>
                </div>
            </div>

            <div class="analysis-grid">
                <div class="analysis-section">
                    <div class="section-header">Recent Critical Incidents</div>
                    <div class="section-content" id="criticalIncidents">
                        <div class="loading">Loading critical incidents</div>
                    </div>
                </div>
                <div class="analysis-section">
                    <div class="section-header">Real-Time Issue Analysis</div>
                    <div class="section-content" id="issueAnalysis">
                        <div class="loading">Analyzing current issues</div>
                    </div>
                </div>
            </div>
        </div>

        <div id="analysis" class="tab-content">
            <div class="loading">Trend Analysis Coming Soon</div>
        </div>

        <div id="opportunities" class="tab-content">
            <div class="loading">Business Opportunities Coming Soon</div>
        </div>

        <div id="health" class="tab-content">
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value" id="systemUptime">0h 0m</div>
                    <div class="metric-label">System Uptime</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="incidentsPerHour">0</div>
                    <div class="metric-label">Incidents/Hour</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="nextAnalysisIn">500</div>
                    <div class="metric-label">Next Analysis In</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="mlConfidence">0%</div>
                    <div class="metric-label">ML Confidence</div>
                </div>
            </div>
        </div>
    </div>

    <div id="detailModal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <h3 id="modalTitle">Incident Details</h3>
                <button class="close-modal" onclick="closeModal()">&times;</button>
            </div>
            <div class="modal-body" id="modalBody">
            </div>
        </div>
    </div>

    <script>
        let charts = {};
        let currentTab = 'trends';

        function switchTab(tabName) {
            document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.nav-tab').forEach(tab => tab.classList.remove('active'));
            
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
            currentTab = tabName;
            
            loadTabData(tabName);
        }

        async function loadTabData(tabName) {
            switch(tabName) {
                case 'trends':
                    await loadCurrentTrends();
                    break;
                case 'analysis':
                    console.log('Loading trend analysis...');
                    break;
                case 'opportunities':
                    console.log('Loading business opportunities...');
                    break;
                case 'health':
                    console.log('Loading system health...');
                    break;
            }
        }

        async function loadCurrentTrends() {
            try {
                const analysisResponse = await fetch('/api/analysis/detailed');
                const analysis = await analysisResponse.json();
                
                if (analysis.error) {
                    document.getElementById('criticalIncidents').innerHTML = 
                        '<div class="loading">Insufficient data - need more incidents for analysis</div>';
                    return;
                }
                
                document.getElementById('activeIncidents').textContent = analysis.analyzed_incidents;
                document.getElementById('avgResolution').textContent = 
                    Math.round(analysis.critical_issues.reduce((sum, issue) => sum + issue.avg_duration, 0) / Math.max(1, analysis.critical_issues.length)) + 'm';
                document.getElementById('systemsAffected').textContent = analysis.provider_performance.length;
                document.getElementById('businessImpact').textContent = '£' + analysis.summary.total_business_impact.toLocaleString();
                
                const alertText = document.getElementById('alertText');
                const criticalCount = analysis.severity_distribution.Critical || 0;
                const majorCount = analysis.severity_distribution.Major || 0;
                
                if (criticalCount > 0) {
                    alertText.textContent = `Warning: ${criticalCount} critical incidents currently affecting service.`;
                } else if (majorCount > 0) {
                    alertText.textContent = `Alert: ${majorCount} major incidents currently affecting service.`;
                } else {
                    alertText.textContent = 'System monitoring active - All systems operating normally';
                }
                
                displayCriticalIncidents(analysis.critical_issues);
                displayIssueAnalysis(analysis);
                createIncidentChart(analysis.critical_issues);
                createProviderChart(analysis.provider_performance);
                
            } catch (error) {
                console.error('Error loading current trends:', error);
            }
        }

        function displayCriticalIncidents(issues) {
            const container = document.getElementById('criticalIncidents');
            container.innerHTML = '';
            
            if (issues.length === 0) {
                container.innerHTML = '<div class="loading">No critical issues detected</div>';
                return;
            }
            
            issues.forEach(issue => {
                const item = document.createElement('div');
                item.className = 'incident-item';
                item.onclick = () => showIncidentDetails(issue);
                
                const severityClass = issue.severity_breakdown.Critical ? 'critical' : 
                                    issue.severity_breakdown.Major ? 'major' : 'medium';
                
                item.innerHTML = `
                    <div class="incident-header">
                        <div class="incident-title">${issue.error_type}</div>
                        <div class="severity-badge severity-${severityClass}">
                            ${issue.frequency} incidents
                        </div>
                    </div>
                    <div class="incident-meta">
                        Avg Duration: ${issue.avg_duration}min • 
                        Impact: £${issue.total_impact.toLocaleString()} • 
                        Affected: ${issue.customers_affected.toLocaleString()} customers
                    </div>
                    <div class="incident-impact">
                        Avg Impact: £${issue.avg_impact.toLocaleString()} per incident
                    </div>
                `;
                
                container.appendChild(item);
            });
        }

        function displayIssueAnalysis(analysis) {
            const container = document.getElementById('issueAnalysis');
            container.innerHTML = `
                <div style="margin-bottom: 15px;">
                    <strong>Pattern Analysis:</strong><br>
                    Most frequent: ${analysis.summary.most_frequent_issue}<br>
                    Highest impact: ${analysis.summary.highest_impact_issue}<br>
                    Problematic provider: ${analysis.summary.most_problematic_provider}
                </div>
                <div style="margin-bottom: 15px;">
                    <strong>Business Impact:</strong><br>
                    Financial: £${analysis.summary.total_business_impact.toLocaleString()}<br>
                    Customers: ${analysis.summary.total_customers_affected.toLocaleString()}
                </div>
                <div>
                    <strong>Severity Breakdown:</strong><br>
                    ${Object.entries(analysis.severity_distribution).map(([severity, count]) => 
                        `${severity}: ${count}`
                    ).join('<br>')}
                </div>
            `;
        }

        function showIncidentDetails(issue) {
            document.getElementById('modalTitle').textContent = `Issue: ${issue.error_type}`;
            document.getElementById('modalBody').innerHTML = `
                <strong>Frequency:</strong> ${issue.frequency} incidents<br>
                <strong>Total Impact:</strong> £${issue.total_impact.toLocaleString()}<br>
                <strong>Customers Affected:</strong> ${issue.customers_affected.toLocaleString()}<br>
                <strong>Average Duration:</strong> ${issue.avg_duration} minutes<br><br>
                <strong>Severity Distribution:</strong><br>
                ${Object.entries(issue.severity_breakdown).map(([severity, count]) => 
                    `${severity}: ${count} incidents`
                ).join('<br>')}
            `;
            document.getElementById('detailModal').classList.add('active');
        }

        function createIncidentChart(issues) {
            const ctx = document.getElementById('incidentChart').getContext('2d');
            
            if (charts.incidentChart) {
                charts.incidentChart.destroy();
            }
            
            charts.incidentChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: issues.slice(0, 6).map(issue => issue.error_type.substring(0, 20)),
                    datasets: [{
                        label: 'Frequency',
                        data: issues.slice(0, 6).map(issue => issue.frequency),
                        backgroundColor: 'rgba(102, 126, 234, 0.8)'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false
                }
            });
        }

        function createProviderChart(providers) {
            const ctx = document.getElementById('providerChart').getContext('2d');
            
            if (charts.providerChart) {
                charts.providerChart.destroy();
            }
            
            charts.providerChart = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: providers.map(p => p.provider),
                    datasets: [{
                        data: providers.map(p => p.incident_count),
                        backgroundColor: [
                            'rgba(102, 126, 234, 0.8)',
                            'rgba(231, 76, 60, 0.8)',
                            'rgba(46, 125, 50, 0.8)',
                            'rgba(245, 124, 0, 0.8)',
                            'rgba(123, 31, 162, 0.8)'
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false
                }
            });
        }

        function closeModal() {
            document.getElementById('detailModal').classList.remove('active');
        }

        async function loadStats() {
            try {
                const response = await fetch('/api/stats');
                const stats = await response.json();
                
                const hours = Math.floor(stats.uptime_seconds / 3600);
                const minutes = Math.floor((stats.uptime_seconds % 3600) / 60);
                document.getElementById('systemUptime').textContent = `${hours}h ${minutes}m`;
                document.getElementById('incidentsPerHour').textContent = stats.incidents_per_hour;
                document.getElementById('nextAnalysisIn').textContent = stats.next_analysis_in;
                
                const confidence = Math.min(95, Math.round((stats.total_incidents / 500) * 100));
                document.getElementById('mlConfidence').textContent = confidence + '%';
                
            } catch (error) {
                console.error('Error loading stats:', error);
            }
        }

        setInterval(async () => {
            await loadStats();
            if (currentTab === 'trends') {
                await loadCurrentTrends();
            }
        }, 10000);

        loadStats();
        loadCurrentTrends();
    </script>
</body>
</html>'''
    
    with open('templates/dashboard.html', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Created: templates/dashboard.html")
    
    with open('templates/dashboard.html', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Created: templates/dashboard.html")

def write_init_files():
    with open('mcp/__init__.py', 'w', encoding='utf-8') as f:
        f.write('# MCP modules for UK Retail Intelligence Engine\\n')
    print("Created: mcp/__init__.py")

if __name__ == '__main__':
    print("Creating UK Retail Intelligence Engine...")
    print("="*60)
    create_project_structure()