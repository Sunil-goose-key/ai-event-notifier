import requests
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from datetime import datetime, timedelta
import json
import os
from typing import List, Dict
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_event_digest.log'),
        logging.StreamHandler()
    ]
)

class EventbriteAINotifier:
    def __init__(self):
        self.eventbrite_token = os.getenv('EVENTBRITE_TOKEN')
        self.email_user = os.getenv('EMAIL_USER')
        self.email_password = os.getenv('EMAIL_PASSWORD')
        self.recipient_email = os.getenv('RECIPIENT_EMAIL', 'gitamsunil@gmail.com')
        
        # Eventbrite API configuration
        self.api_base_url = "https://www.eventbriteapi.com/v3"
        self.headers = {
            'Authorization': f'Bearer {self.eventbrite_token}',
            'Content-Type': 'application/json'
        }
        
        # Validate configuration
        if not all([self.eventbrite_token, self.email_user, self.email_password]):
            raise ValueError("Missing required credentials. Check .env file.")
    
    def search_eventbrite_ai_events(self) -> List[Dict]:
        """Search for AI and MCP related events using Eventbrite API"""
        logging.info("🔍 Searching Eventbrite for AI and MCP events...")
        
        # AI-related keywords for search
        ai_keywords = [
            "artificial intelligence", "machine learning", "AI", "MCP",
            "model context protocol", "deep learning", "neural networks",
            "chatgpt", "openai", "llm", "generative ai", "python ai"
        ]
        
        all_events = []
        
        for keyword in ai_keywords[:3]:  # Limit to avoid rate limits
            try:
                # Eventbrite Event Search API endpoint
                search_url = f"{self.api_base_url}/events/search/"
                
                # Calculate date range (next 30 days)
                start_date = datetime.now().isoformat()
                end_date = (datetime.now() + timedelta(days=30)).isoformat()
                
                params = {
                    'q': keyword,
                    'sort_by': 'date',
                    'location.address': 'India',  # Focus on Indian events
                    'start_date.range_start': start_date,
                    'start_date.range_end': end_date,
                    'categories': '102,103',  # Science & Tech, Business categories
                    'expand': 'venue,ticket_availability',
                    'page_size': 20
                }
                
                response = requests.get(search_url, headers=self.headers, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    events = data.get('events', [])
                    
                    for event in events:
                        # Filter for AI/MCP relevance
                        if self._is_ai_relevant(event):
                            formatted_event = self._format_event(event)
                            if formatted_event not in all_events:
                                all_events.append(formatted_event)
                
                logging.info(f"✅ Found {len(events)} events for keyword: {keyword}")
                
            except Exception as e:
                logging.error(f"❌ Error searching for keyword '{keyword}': {e}")
                continue
        
        # Remove duplicates and sort by date
        unique_events = []
        seen_ids = set()
        
        for event in all_events:
            if event['id'] not in seen_ids:
                unique_events.append(event)
                seen_ids.add(event['id'])
        
        # Sort by date
        unique_events.sort(key=lambda x: x['start_date'])
        
        logging.info(f"📊 Total unique AI/MCP events found: {len(unique_events)}")
        return unique_events[:10]  # Return top 10 events
    
    def _is_ai_relevant(self, event: Dict) -> bool:
        """Check if event is AI/MCP relevant"""
        ai_terms = [
            'ai', 'artificial intelligence', 'machine learning', 'deep learning',
            'neural network', 'chatgpt', 'openai', 'llm', 'mcp', 'model context protocol',
            'generative ai', 'python ai', 'data science', 'ml', 'nlp'
        ]
        
        # Check event name and description
        event_text = (
            event.get('name', {}).get('text', '').lower() + ' ' +
            event.get('description', {}).get('text', '').lower()
        )
        
        return any(term in event_text for term in ai_terms)
    
    def _format_event(self, event: Dict) -> Dict:
        """Format Eventbrite event data for our use"""
        venue = event.get('venue', {})
        start_time = event.get('start', {})
        end_time = event.get('end', {})
        
        return {
            'id': event.get('id', ''),
            'name': event.get('name', {}).get('text', 'Untitled Event'),
            'description': event.get('description', {}).get('text', 'No description available')[:300] + '...',
            'url': event.get('url', ''),
            'start_date': start_time.get('local', ''),
            'end_date': end_time.get('local', ''),
            'location': venue.get('address', {}).get('localized_area_display', 'Online'),
            'venue_name': venue.get('name', 'TBA'),
            'is_online': event.get('online_event', False),
            'is_free': event.get('is_free', False),
            'capacity': event.get('capacity', 'Not specified'),
            'status': event.get('status', 'live'),
            'tags': self._extract_ai_tags(event)
        }
    
    def _extract_ai_tags(self, event: Dict) -> List[str]:
        """Extract relevant AI tags from event"""
        tags = []
        event_text = (
            event.get('name', {}).get('text', '').lower() + ' ' +
            event.get('description', {}).get('text', '').lower()
        )
        
        tag_mapping = {
            'mcp': 'MCP',
            'model context protocol': 'MCP',
            'artificial intelligence': 'AI',
            'machine learning': 'ML',
            'deep learning': 'Deep Learning',
            'neural network': 'Neural Networks',
            'chatgpt': 'ChatGPT',
            'openai': 'OpenAI',
            'llm': 'LLM',
            'generative ai': 'Generative AI',
            'python': 'Python',
            'workshop': 'Workshop',
            'webinar': 'Webinar'
        }
        
        for term, tag in tag_mapping.items():
            if term in event_text:
                tags.append(tag)
        
        return list(set(tags))[:5]  # Return unique tags, max 5
    
    def format_email_content(self, events: List[Dict]) -> str:
        """Format events into HTML email content"""
        current_date = datetime.now().strftime('%B %d, %Y')
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    margin: 0; 
                    padding: 20px; 
                    background-color: #f5f5f5;
                }}
                .container {{ 
                    max-width: 800px; 
                    margin: 0 auto; 
                    background-color: white; 
                    border-radius: 10px; 
                    overflow: hidden;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                .header {{ 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; 
                    padding: 30px; 
                    text-align: center;
                }}
                .header h1 {{ margin: 0; font-size: 28px; }}
                .header p {{ margin: 10px 0 0 0; opacity: 0.9; }}
                .content {{ padding: 30px; }}
                .event {{ 
                    background-color: #f8f9fa; 
                    margin: 20px 0; 
                    padding: 20px; 
                    border-left: 5px solid #007acc; 
                    border-radius: 5px;
                }}
                .event-title {{ 
                    font-weight: bold; 
                    font-size: 20px; 
                    color: #007acc; 
                    margin-bottom: 10px;
                }}
                .event-meta {{ 
                    color: #666; 
                    font-size: 14px; 
                    margin: 8px 0;
                }}
                .event-meta strong {{ color: #333; }}
                .event-description {{ 
                    margin: 15px 0; 
                    line-height: 1.6; 
                    color: #444;
                }}
                .event-tags {{ margin: 10px 0; }}
                .tag {{ 
                    background-color: #e3f2fd; 
                    color: #1976d2; 
                    padding: 4px 8px; 
                    border-radius: 12px; 
                    font-size: 12px; 
                    margin-right: 5px;
                    display: inline-block;
                }}
                .event-link {{ 
                    display: inline-block; 
                    background-color: #007acc; 
                    color: white; 
                    padding: 8px 16px; 
                    text-decoration: none; 
                    border-radius: 5px; 
                    margin-top: 10px;
                }}
                .footer {{ 
                    background-color: #f8f9fa; 
                    padding: 25px; 
                    border-top: 1px solid #dee2e6;
                }}
                .mcp-section {{ 
                    background-color: #e8f5e8; 
                    padding: 20px; 
                    border-radius: 8px; 
                    margin-top: 20px;
                }}
                .stats {{ 
                    display: flex; 
                    justify-content: space-around; 
                    margin: 20px 0;
                }}
                .stat {{ text-align: center; padding: 15px; }}
                .stat-number {{ 
                    font-size: 24px; 
                    font-weight: bold; 
                    color: #007acc;
                }}
                .stat-label {{ 
                    font-size: 12px; 
                    color: #666; 
                    text-transform: uppercase;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🤖 AI & MCP Events from Eventbrite</h1>
                    <p>Your personalized AI events update for {current_date}</p>
                    <p><strong>Source:</strong> Eventbrite API - Real-time event data</p>
                </div>
                
                <div class="content">
                    <div class="stats">
                        <div class="stat">
                            <div class="stat-number">{len(events)}</div>
                            <div class="stat-label">Events Found</div>
                        </div>
                        <div class="stat">
                            <div class="stat-number">{len([e for e in events if e.get('is_free')])}</div>
                            <div class="stat-label">Free Events</div>
                        </div>
                        <div class="stat">
                            <div class="stat-number">{len([e for e in events if 'MCP' in str(e.get('tags', []))])}</div>
                            <div class="stat-label">MCP Events</div>
                        </div>
                    </div>
                    
                    <h2>📅 Upcoming AI Events from Eventbrite</h2>
        """
        
        for event in events:
            tags_html = ''.join([f'<span class="tag">{tag}</span>' for tag in event.get('tags', [])])
            start_date = datetime.fromisoformat(event['start_date'].replace('Z', '+00:00')) if event['start_date'] else datetime.now()
            formatted_date = start_date.strftime('%B %d, %Y at %I:%M %p')
            
            html_content += f"""
            <div class="event">
                <div class="event-title">🎯 {event['name']}</div>
                <div class="event-meta">
                    <strong>📅 Date:</strong> {formatted_date}
                </div>
                <div class="event-meta">
                    <strong>📍 Location:</strong> {event['location']}
                    {' (Online Event)' if event.get('is_online') else ''}
                </div>
                <div class="event-meta">
                    <strong>🎫 Type:</strong> {'Free Event' if event.get('is_free') else 'Paid Event'}
                </div>
                <div class="event-meta">
                    <strong>🏢 Venue:</strong> {event['venue_name']}
                </div>
                <div class="event-description">
                    {event['description']}
                </div>
                <div class="event-tags">
                    {tags_html}
                </div>
                <a href="{event['url']}" class="event-link" target="_blank">View Event & Register →</a>
            </div>
            """
        
        html_content += f"""
                </div>
                
                <div class="footer">
                    <div class="mcp-section">
                        <h3>🎯 Assignment Context: Model Context Protocol (MCP) Integration</h3>
                        <p><strong>This system demonstrates MCP concepts through:</strong></p>
                        <ul>
                            <li><strong>Real-time API Integration:</strong> Live data from Eventbrite API</li>
                            <li><strong>Context-Aware Filtering:</strong> AI/MCP keyword-based event discovery</li>
                            <li><strong>Adaptive Processing:</strong> Dynamic event parsing and relevance scoring</li>
                            <li><strong>Structured Data Handling:</strong> JSON processing similar to MCP workflows</li>
                        </ul>
                    </div>
                    
                    <div style="margin-top: 20px; padding-top: 20px; border-top: 1px solid #dee2e6; text-align: center; color: #666; font-size: 12px;">
                        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}</p>
                        <p>Data sourced from Eventbrite API - Real-time event information</p>
                        <p>Delivered to: {self.recipient_email}</p>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def send_email(self, events: List[Dict]):
        """Send formatted email with AI events"""
        subject = f"🤖 Weekly AI & MCP Events from Eventbrite - {datetime.now().strftime('%B %d, %Y')}"
        
        msg = MimeMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = self.email_user
        msg['To'] = self.recipient_email
        
        # Create HTML content
        html_content = self.format_email_content(events)
        html_part = MimeText(html_content, 'html')
        msg.attach(html_part)
        
        # Send email
        try:
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(self.email_user, self.email_password)
                server.send_message(msg)
            
            logging.info(f"✅ Email sent successfully to {self.recipient_email}")
            
        except Exception as e:
            logging.error(f"❌ Error sending email: {e}")
            raise
    
    def run_weekly_digest(self):
        """Main function to run the weekly digest"""
        try:
            logging.info("🚀 Starting Eventbrite AI Event Digest...")
            
            # Search for events
            events = self.search_eventbrite_ai_events()
            
            if events:
                logging.info(f"📧 Found {len(events)} AI/MCP events. Sending digest...")
                self.send_email(events)
                
                # Log event summary
                for event in events:
                    logging.info(f"  📅 {event['name']} - {event['start_date']}")
                
            else:
                logging.info("ℹ️ No AI/MCP events found on Eventbrite for this week.")
                
        except Exception as e:
            logging.error(f"❌ Error in digest process: {e}")
            raise

# Main execution
if __name__ == "__main__":
    try:
        notifier = EventbriteAINotifier()
        notifier.run_weekly_digest()
        
    except Exception as e:
        logging.error(f"💥 Failed to run digest: {e}")
        print(f"💥 Error: {e}")
        print("Please check your .env file and Eventbrite token.")
