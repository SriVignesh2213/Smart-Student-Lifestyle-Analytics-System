"""
MCP Agent Integration for AI-Powered Insights
Provides reasoning-based responses and personalized recommendations using language models
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class StudentProfile:
    """Represents a student's lifestyle profile"""
    roll_no: str
    age: int
    department: str
    year_of_study: int
    features: Dict[str, float]
    lifestyle_score: float
    burnout_risk: float
    digital_addiction_score: float
    productivity_index: float


class MCPAgent:
    """
    MCP (Model Context Protocol) Agent for AI-powered insights
    Provides reasoning-based recommendations based on student data and predictions
    """
    
    def __init__(self):
        """Initialize MCP agent"""
        self.conversation_history = []
        self.system_context = self._build_system_context()
    
    def _build_system_context(self) -> str:
        """Build system context for reasoning"""
        return """
        You are an expert AI assistant specialized in student wellness and lifestyle optimization.
        Your role is to:
        1. Analyze student lifestyle patterns
        2. Identify root causes of burnout and low productivity
        3. Provide actionable, personalized recommendations
        4. Consider academic, health, and social factors
        5. Reference data-driven insights from ML models
        
        Guidelines:
        - Be empathetic and supportive in your tone
        - Provide specific, measurable recommendations
        - Consider context (year of study, department, current stress levels)
        - Suggest realistic changes that can be implemented immediately
        - Reference peer benchmarks when relevant
        """
    
    def analyze_query(self, query: str, student_profile: Optional[StudentProfile] = None) -> Dict:
        """
        Analyze user query and generate contextualized response
        
        Args:
            query: User's question or concern
            student_profile: Student's data profile (optional)
            
        Returns:
            Dictionary with analysis and recommendations
        """
        logger.info(f"🔍 Analyzing query: {query}")
        
        # Identify query type
        query_type = self._classify_query(query)
        
        # Generate response based on type
        if query_type == 'performance':
            response = self._handle_performance_query(query, student_profile)
        elif query_type == 'health':
            response = self._handle_health_query(query, student_profile)
        elif query_type == 'stress':
            response = self._handle_stress_query(query, student_profile)
        elif query_type == 'productivity':
            response = self._handle_productivity_query(query, student_profile)
        else:
            response = self._handle_general_query(query, student_profile)
        
        # Log interaction
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'query_type': query_type,
            'response': response,
            'student_id': student_profile.roll_no if student_profile else None
        })
        
        return response
    
    def _classify_query(self, query: str) -> str:
        """Classify query into categories"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['score', 'gpa', 'performance', 'academic']):
            return 'performance'
        elif any(word in query_lower for word in ['sleep', 'tired', 'health', 'physical']):
            return 'health'
        elif any(word in query_lower for word in ['stress', 'anxious', 'overwhelmed', 'pressure']):
            return 'stress'
        elif any(word in query_lower for word in ['productivity', 'focus', 'concentration', 'efficient']):
            return 'productivity'
        else:
            return 'general'
    
    def _handle_performance_query(self, query: str, profile: Optional[StudentProfile]) -> Dict:
        """Handle academic performance related queries"""
        
        insights = {
            'category': 'Academic Performance',
            'insights': [],
            'recommendations': [],
            'action_items': []
        }
        
        if profile:
            if profile.lifestyle_score < 4:
                insights['insights'].append(
                    "Your lifestyle score indicates irregular sleep/study patterns which directly "
                    "impact academic performance."
                )
                insights['recommendations'].append(
                    "Establish consistent sleep schedule (10:30 PM - 6:30 AM)"
                )
                insights['action_items'].append(
                    "Week 1: Set alarm 1 hour earlier each day"
                )
            
            if profile.productivity_index < 2:
                insights['insights'].append(
                    "Your productivity index suggests concentration issues, likely due to "
                    "high screen time or inadequate breaks."
                )
                insights['recommendations'].append(
                    "Implement 25-minute Pomodoro sessions with 5-minute breaks"
                )
                insights['action_items'].append(
                    "Day 1: Install Forest app to track focus sessions"
                )
            
            if profile.features.get('study_hours', 0) < 3:
                insights['insights'].append(
                    "Study hours below peer average. Quality study time needed for GPA improvement."
                )
                insights['recommendations'].append(
                    "Increase focused study to 4-5 hours daily with strategic breaks"
                )
                insights['action_items'].append(
                    "Create weekly study schedule with allocated subjects"
                )
        else:
            insights['insights'].append(
                "To get personalized academic insights, please share your lifestyle data."
            )
            insights['recommendations'].append(
                "Fill the prediction form with your daily habits (sleep, study, screen time, etc.)"
            )
        
        return insights
    
    def _handle_health_query(self, query: str, profile: Optional[StudentProfile]) -> Dict:
        """Handle health and sleep related queries"""
        
        insights = {
            'category': 'Health & Wellness',
            'insights': [],
            'recommendations': [],
            'action_items': []
        }
        
        if profile:
            if profile.features.get('sleep_hours', 0) < 6:
                insights['insights'].append(
                    f"You're averaging {profile.features.get('sleep_hours', 0):.1f} hours of sleep, "
                    "which is below the recommended 7-8 hours for students."
                )
                insights['recommendations'].append(
                    "Gradually increase sleep by 15 minutes per night until reaching 7-8 hours"
                )
                insights['action_items'].append(
                    "Tonight: Go to bed 15 minutes earlier than usual"
                )
            
            if profile.features.get('screen_time', 0) > 7:
                insights['insights'].append(
                    "High screen time is affecting your sleep quality and eye health."
                )
                insights['recommendations'].append(
                    "Implement 'digital sunset' - no screens 30 minutes before sleep"
                )
                insights['action_items'].append(
                    "Enable blue light filter on all devices after 8 PM"
                )
            
            if profile.features.get('exercise', 0) < 2:
                insights['insights'].append(
                    "Regular exercise is crucial for energy levels and stress management."
                )
                insights['recommendations'].append(
                    "Start with 20-minute walks daily, progressing to gym 3-4 times/week"
                )
                insights['action_items'].append(
                    "Tomorrow: 20-minute walk on campus"
                )
        else:
            insights['insights'].append(
                "General health recommendations: Aim for 7-8 hours sleep, 30+ minutes exercise daily"
            )
            insights['recommendations'].append(
                "Track your sleep and exercise patterns for better insights"
            )
        
        return insights
    
    def _handle_stress_query(self, query: str, profile: Optional[StudentProfile]) -> Dict:
        """Handle stress and burnout related queries"""
        
        insights = {
            'category': 'Stress Management',
            'insights': [],
            'recommendations': [],
            'action_items': [],
            'immediate_relief': []
        }
        
        if profile:
            if profile.burnout_risk > 5:
                insights['insights'].append(
                    "High burnout risk detected. Immediate intervention needed."
                )
                insights['immediate_relief'].extend([
                    "Take a 10-minute walk outside",
                    "Practice 4-7-8 breathing (4 in, 7 hold, 8 out)",
                    "Connect with a friend or counselor"
                ])
                insights['recommendations'].extend([
                    "Daily 10-minute meditation practice",
                    "Weekly mental health check-ins with counselor",
                    "Prioritize 3-4 hours minimum sleep tonight"
                ])
                insights['action_items'].append(
                    "Week 1: Download Calm or Headspace for guided meditation"
                )
            
            if profile.features.get('stress_level', 0) > 7:
                insights['insights'].append(
                    "Your stress level is elevated. Implement stress-reduction techniques."
                )
                insights['recommendations'].append(
                    "Practice yoga, meditation, or progressive muscle relaxation"
                )
                insights['action_items'].append(
                    "Today: Find a quiet space for 5-minute breathing exercise"
                )
        else:
            insights['immediate_relief'] = [
                "Box breathing: 4-4-4-4 pattern",
                "Progressive muscle relaxation",
                "5-minute mindfulness exercise"
            ]
            insights['recommendations'] = [
                "Build regular exercise routine",
                "Practice daily meditation (10 minutes)",
                "Maintain work-life balance"
            ]
        
        return insights
    
    def _handle_productivity_query(self, query: str, profile: Optional[StudentProfile]) -> Dict:
        """Handle productivity and focus related queries"""
        
        insights = {
            'category': 'Productivity Enhancement',
            'insights': [],
            'recommendations': [],
            'action_items': [],
            'tools': []
        }
        
        if profile:
            if profile.features.get('concentration', 0) < 2:
                insights['insights'].append(
                    "Concentration is below optimal levels. Primary causes: screen time, "
                    "inadequate breaks, or sleep deprivation."
                )
                insights['recommendations'].extend([
                    "Implement Pomodoro Technique: 25 min work, 5 min break",
                    "Reduce social media usage to scheduled times only",
                    "Study in distraction-free environment"
                ])
                insights['tools'].extend([
                    "Forest App (focus timer with gamification)",
                    "Cold Turkey (website blocker)",
                    "Focus@Will (concentration music)"
                ])
            
            if profile.features.get('social_media_usage', 0) > 2:
                insights['insights'].append(
                    "High social media usage is fragmenting your attention."
                )
                insights['recommendations'].append(
                    "Limit social media to 1 hour daily, preferably after study sessions"
                )
                insights['action_items'].append(
                    "Remove social media apps from home screen"
                )
            
            if profile.features.get('time_management', 0) < 2:
                insights['insights'].append(
                    "Time management skills need improvement for better productivity."
                )
                insights['recommendations'].extend([
                    "Create weekly schedule with time blocks",
                    "Prioritize tasks using Eisenhower Matrix (Important vs Urgent)",
                    "Review and adjust weekly"
                ])
                insights['tools'].append("Todoist or Notion for task management")
        else:
            insights['recommendations'] = [
                "Use Pomodoro Technique for focused work",
                "Time-block your calendar",
                "Minimize digital distractions"
            ]
            insights['tools'] = [
                "Pomodoro Timer Apps",
                "Task Management: Todoist, Notion, Microsoft To-Do",
                "Focus Music: Focus@Will, Noisli"
            ]
        
        return insights
    
    def _handle_general_query(self, query: str, profile: Optional[StudentProfile]) -> Dict:
        """Handle general queries"""
        
        return {
            'category': 'General Inquiry',
            'response': (
                "This is a general query about student wellness. "
                "For personalized insights, please use one of our specific categories: "
                "Academic Performance, Health & Wellness, Stress Management, or Productivity."
            ),
            'suggestions': [
                "Ask about your academic performance",
                "Ask about sleep and health concerns",
                "Ask about stress and burnout",
                "Ask about productivity and focus"
            ]
        }
    
    def get_peer_benchmark(self, metric: str, student_value: float, peer_data: List[float]) -> Dict:
        """
        Compare student metric against peer group
        
        Args:
            metric: Name of metric
            student_value: Student's value
            peer_data: List of peer values
            
        Returns:
            Benchmark comparison dictionary
        """
        peer_mean = np.mean(peer_data)
        peer_std = np.std(peer_data)
        percentile = (np.sum(np.array(peer_data) <= student_value) / len(peer_data)) * 100
        
        return {
            'metric': metric,
            'student_value': student_value,
            'peer_mean': peer_mean,
            'peer_std': peer_std,
            'percentile': percentile,
            'comparison': 'Above average' if student_value > peer_mean else 'Below average',
            'gap': abs(student_value - peer_mean)
        }
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history for audit trail"""
        return self.conversation_history
    
    def save_session(self, filepath: str):
        """Save conversation session to file"""
        with open(filepath, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)
        logger.info(f"✅ Session saved to {filepath}")
    
    def generate_wellness_report(self, profile: StudentProfile) -> str:
        """Generate comprehensive wellness report"""
        
        report = f"""
        ╔═══════════════════════════════════════════════════════════════╗
        ║        STUDENT WELLNESS & LIFESTYLE ANALYSIS REPORT           ║
        ╚═══════════════════════════════════════════════════════════════╝
        
        📊 STUDENT INFORMATION
        ├─ Roll No: {profile.roll_no}
        ├─ Department: {profile.department}
        ├─ Year of Study: Year {profile.year_of_study}
        └─ Age: {profile.age}
        
        📈 LIFESTYLE METRICS
        ├─ Lifestyle Score: {profile.lifestyle_score:.2f}/10
        ├─ Burnout Risk: {profile.burnout_risk:.2f}/10
        ├─ Digital Addiction: {profile.digital_addiction_score:.2f}/10
        └─ Productivity Index: {profile.productivity_index:.2f}/10
        
        🎯 KEY INSIGHTS
        {self._generate_key_insights(profile)}
        
        💡 RECOMMENDATIONS
        {self._generate_recommendations(profile)}
        
        ✅ ACTION PLAN (Next 7 Days)
        {self._generate_action_plan(profile)}
        
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return report
    
    def _generate_key_insights(self, profile: StudentProfile) -> str:
        """Generate key insights from profile"""
        insights = []
        
        if profile.lifestyle_score > 7:
            insights.append("✅ Excellent lifestyle habits - continue current routine")
        elif profile.lifestyle_score > 5:
            insights.append("⚠️  Good but room for improvement")
        else:
            insights.append("🔴 Significant lifestyle changes needed")
        
        if profile.burnout_risk > 6:
            insights.append("🔴 HIGH BURNOUT RISK - Immediate action required")
        
        return "\n        ".join([f"├─ {insight}" for insight in insights])
    
    def _generate_recommendations(self, profile: StudentProfile) -> str:
        """Generate recommendations"""
        recommendations = []
        
        if profile.features.get('sleep_hours', 0) < 7:
            recommendations.append("Increase sleep to 7-8 hours nightly")
        if profile.features.get('screen_time', 0) > 7:
            recommendations.append("Reduce screen time by 2 hours daily")
        if profile.features.get('exercise', 0) < 2:
            recommendations.append("Add 30+ minutes exercise 3x/week")
        if profile.features.get('stress_level', 0) > 6:
            recommendations.append("Practice daily meditation (10 minutes)")
        
        return "\n        ".join([f"├─ {rec}" for rec in recommendations])
    
    def _generate_action_plan(self, profile: StudentProfile) -> str:
        """Generate 7-day action plan"""
        actions = [
            "Day 1-2: Establish consistent sleep schedule",
            "Day 3-4: Implement focus breaks and reduce screen time",
            "Day 5-6: Start light exercise routine",
            "Day 7: Review progress and adjust as needed"
        ]
        
        return "\n        ".join([f"├─ {action}" for action in actions])


# Example usage function
def create_mcp_agent() -> MCPAgent:
    """Factory function to create MCP agent"""
    return MCPAgent()
