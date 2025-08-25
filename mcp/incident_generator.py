#!/usr/bin/env python3

import random
import json
from datetime import datetime, timedelta
from openai import OpenAI
import os

class IncidentGenerator:
    def __init__(self):
        print("INIT: Starting IncidentGenerator initialization")
        print("INIT: Attempting OpenAI import and initialization")
        
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        print("INIT: OpenAI client initialized successfully")
        
        self.error_types = [
            "API Rate Limit Exceeded", "Database Connection Timeout", "Service Discovery Error",
            "Authentication Service Down", "Payment Gateway Failure", "Cache Invalidation Error",
            "Load Balancer Malfunction", "CDN Origin Server Error", "SSL Certificate Expired",
            "Memory Leak Detected", "Disk Space Exhaustion", "Network Latency Spike",
            "Third-party Integration Failure", "Data Corruption Detected", "Session Management Error",
            "Search Index Rebuild Required", "Queue Processing Backlog", "Configuration Drift Detected",
            "Security Breach Attempt", "Compliance Violation Detected", "Backup System Failure",
            "Monitoring Alert Storm", "Container Orchestration Failure", "DNS Resolution Failure",
            "Email Service Disruption", "File Upload Service Error", "User Session Timeout",
            "Product Search Timeout", "Inventory Sync Failure", "Order Processing Delay",
            "Customer Account Lockout", "Returns Portal Down", "Recommendation Engine Error",
            "Microservice Circuit Breaker Triggered", "Kafka Message Queue Overload", "Redis Cache Cluster Down",
            "Elasticsearch Index Corruption", "MongoDB Replica Set Failure", "PostgreSQL Deadlock Storm",
            "Kubernetes Pod Eviction", "Docker Container Registry Unavailable", "Service Mesh Proxy Error",
            "GraphQL Schema Validation Failure", "REST API Endpoint Deprecated", "WebSocket Connection Drop",
            "gRPC Service Unavailable", "Message Broker Partition Failure", "Event Stream Processing Lag",
            "CQRS Command Handler Timeout", "Event Sourcing Snapshot Corruption", "Saga Transaction Rollback",
            "Distributed Lock Contention", "Eventual Consistency Violation", "CAP Theorem Partition Tolerance",
            "Blockchain Network Congestion", "Smart Contract Gas Exhaustion", "Cryptocurrency Wallet Sync Failure",
            "OAuth Token Expiration", "SAML Assertion Validation Error", "JWT Signature Verification Failed",
            "Multi-Factor Authentication Bypass", "Password Hash Algorithm Weakness", "Session Fixation Attack",
            "Cross-Site Request Forgery", "SQL Injection Attempt Blocked", "NoSQL Injection Vector Found",
            "XML External Entity Attack", "Deserialization Vulnerability", "Buffer Overflow Detection",
            "Race Condition Exploit", "Time-of-Check-Time-of-Use Bug", "Integer Overflow Exception",
            "Null Pointer Dereference", "Stack Overflow Crash", "Heap Memory Corruption",
            "Garbage Collection Pause Spike", "JIT Compilation Failure", "Assembly Code Injection",
            "Virtual Machine Escape Attempt", "Hypervisor Security Breach", "Container Runtime Vulnerability",
            "Kernel Module Crash", "Device Driver Malfunction", "Hardware Thermal Throttling",
            "CPU Cache Coherency Issue", "Memory Bus Error", "Storage I/O Subsystem Failure",
            "Network Interface Card Reset", "Switch Port Flapping", "Router BGP Convergence Delay",
            "OSPF Neighbor Down", "VLAN Configuration Mismatch", "Spanning Tree Protocol Loop",
            "Quality of Service Policy Violation", "Bandwidth Utilization Ceiling", "Packet Loss Threshold Exceeded",
            "Jitter Buffer Underrun", "Echo Cancellation Failure", "Codec Transcoding Error",
            "SIP Registration Timeout", "RTP Stream Interruption", "DTMF Tone Detection Failure",
            "Video Streaming Bitrate Drop", "Audio Synchronization Drift", "Subtitle Rendering Lag",
            "Content Delivery Network Edge Failure", "Origin Server Overload", "Cache Hit Ratio Degradation",
            "Image Optimization Service Down", "Video Transcoding Queue Stalled", "Live Stream Encoder Crash",
            "Recommendation Algorithm Bias", "Machine Learning Model Drift", "Neural Network Training Divergence",
            "Feature Store Inconsistency", "Data Pipeline Transformation Error", "ETL Process Memory Exhaustion",
            "Data Lake Query Timeout", "Data Warehouse Schema Migration", "Business Intelligence Report Failure"
        ]
        
        self.service_providers = [
            "AWS Retail Solutions", "Azure Commerce Platform", "Google Cloud Retail",
            "Shopify Plus Enterprise", "Magento Commerce Cloud", "SAP Commerce Cloud",
            "Oracle Retail Solutions", "Salesforce Commerce Cloud", "Adobe Experience Platform",
            "IBM Watson Commerce", "BigCommerce Enterprise", "WooCommerce VIP"
        ]
        
        self.severities = ["Critical", "Major", "Medium", "Minor"]
        self.locations = [
            "London", "Manchester", "Birmingham", "Glasgow", "Liverpool", 
            "Edinburgh", "Bristol", "Leeds", "Sheffield", "Cardiff"
        ]
        
        self.postcodes = [
            "SW1A 1AA", "M1 1AA", "B1 1AA", "G1 1AA", "L1 1AA",
            "EH1 1AA", "BS1 1AA", "LS1 1AA", "S1 1AA", "CF1 1AA"
        ]
        
        print("INIT: IncidentGenerator initialization completed")
    
    def generate_ai_description(self, error_type, service_provider, severity, business_impact, customers_affected, duration_minutes):
        """Use AI to generate detailed incident description"""
        
        prompt = f"""Generate a detailed technical incident description (200-300 words) for:

Error Type: {error_type}
Service Provider: {service_provider}
Severity: {severity}
Duration: {duration_minutes} minutes
Customers Affected: {customers_affected:,}
Business Impact: £{business_impact:,}

Write a professional incident report description that includes:
- What happened and when
- Technical details about the failure
- Customer impact specifics  
- Current status and recovery efforts
- Business implications

Use technical language appropriate for enterprise incident management."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"AI Description Error: {e}")
            return f"System incident in {service_provider} affecting {customers_affected:,} customers with {error_type.lower()}."
    
    def generate_ai_root_cause(self, error_type, service_provider):
        """Use AI to generate detailed root cause analysis"""
        
        prompt = f"""Generate a detailed technical root cause analysis (150-250 words) for:

Error Type: {error_type}
Service Provider: {service_provider}

Explain the underlying technical reason this incident occurred. Include:
- Primary cause of the failure
- Contributing factors
- Why existing safeguards failed
- Technical details about the failure mode
- System dependencies that were affected

Write in technical language for engineering teams."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=350,
                temperature=0.6
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"AI Root Cause Error: {e}")
            return f"Technical issue in {service_provider} caused by {error_type.lower()}."
    
    def generate_ai_recovery_actions(self, error_type, service_provider, severity):
        """Use AI to generate recovery actions"""
        
        prompt = f"""Generate 3-5 specific recovery actions for resolving:

Error Type: {error_type}
Service Provider: {service_provider}
Severity: {severity}

List concrete technical steps the engineering team should take to resolve this incident."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.5
            )
            actions_text = response.choices[0].message.content.strip()
            # Split into list and clean up
            actions = [action.strip('- ').strip() for action in actions_text.split('\n') if action.strip()]
            return actions[:5]  # Limit to 5 actions
        except Exception as e:
            print(f"AI Recovery Actions Error: {e}")
            return ["System restart", "Check logs", "Contact support"]
    
    def generate_dynamic_incident(self):
        print("GEN: Starting incident generation")
        
        error_type = random.choice(self.error_types)
        service_provider = random.choice(self.service_providers)
        severity = random.choice(self.severities)
        location = random.choice(self.locations)
        postcode = random.choice(self.postcodes)
        
        print(f"GEN: Selected {error_type} on {service_provider}")
        
        # Generate realistic metrics based on severity
        if severity == "Critical":
            business_impact = random.randint(50000, 500000)
            customers_affected = random.randint(5000, 50000)
            duration_minutes = random.randint(30, 600)
        elif severity == "Major":
            business_impact = random.randint(10000, 100000)
            customers_affected = random.randint(1000, 10000)
            duration_minutes = random.randint(15, 300)
        elif severity == "Medium":
            business_impact = random.randint(1000, 25000)
            customers_affected = random.randint(100, 2000)
            duration_minutes = random.randint(10, 120)
        else:  # Minor
            business_impact = random.randint(100, 5000)
            customers_affected = random.randint(10, 500)
            duration_minutes = random.randint(5, 60)
        
        # Generate AI content
        description = self.generate_ai_description(error_type, service_provider, severity, business_impact, customers_affected, duration_minutes)
        root_cause = self.generate_ai_root_cause(error_type, service_provider)
        recovery_actions = self.generate_ai_recovery_actions(error_type, service_provider, severity)
        
        # Generate affected systems based on error type
        affected_systems = [f"{service_provider} System"]
        if "Database" in error_type:
            affected_systems.append("Database Cluster")
        if "API" in error_type:
            affected_systems.append("API Gateway")
        if "Cache" in error_type:
            affected_systems.append("Redis Cache")
        
        incident = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "service_provider": service_provider,
            "severity": severity,
            "business_impact": business_impact,
            "customers_affected": customers_affected,
            "duration_minutes": duration_minutes,
            "location": location,
            "postcode": postcode,
            "currency": "GBP",
            "title": f"{error_type} - {service_provider}",
            "description": description,
            "root_cause": root_cause,
            "affected_systems": affected_systems,
            "recovery_actions": recovery_actions
        }
        
        print("GEN: Incident generated successfully")
        return incident
