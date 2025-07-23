"""
Summary Aggregation Module

This module handles building the final comprehensive summary from chunk summaries:
1. Combine chunk summaries into structured sections
2. Generate Abstract-Style Overview
3. Extract Key Insights
4. Create FAQs from content
5. Identify Hidden Gems & Actionable Tips
6. Export to text and optionally PDF
"""

import logging
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SummaryAggregator:
    """Handles aggregation and structuring of chunk summaries"""
    
    def __init__(self):
        """Initialize the aggregator"""
        self.chunk_summaries = []
        self.final_summary = {}
    
    def analyze_content_themes(self, summaries: List[str]) -> Dict[str, List[str]]:
        """
        Analyze summaries to identify common themes and topics
        
        Args:
            summaries: List of chunk summaries
            
        Returns:
            Dictionary mapping themes to related content
        """
        themes = {
            'technical': [],
            'business': [],
            'educational': [],
            'practical': [],
            'conceptual': []
        }
        
        # Keywords for theme classification
        theme_keywords = {
            'technical': ['algorithm', 'code', 'programming', 'software', 'system', 'technology', 
                         'data', 'model', 'framework', 'API', 'database', 'architecture'],
            'business': ['market', 'business', 'strategy', 'revenue', 'profit', 'customer', 
                        'company', 'industry', 'competition', 'growth', 'investment'],
            'educational': ['learn', 'understand', 'explain', 'concept', 'theory', 'principle', 
                           'knowledge', 'study', 'research', 'analysis', 'method'],
            'practical': ['how to', 'step', 'process', 'implement', 'apply', 'use', 'practice', 
                         'example', 'demonstration', 'tutorial', 'guide'],
            'conceptual': ['idea', 'concept', 'philosophy', 'approach', 'perspective', 'vision', 
                          'future', 'innovation', 'creative', 'thinking']
        }
        
        for summary in summaries:
            summary_lower = summary.lower()
            
            # Count keyword matches for each theme
            theme_scores = {}
            for theme, keywords in theme_keywords.items():
                score = sum(1 for keyword in keywords if keyword in summary_lower)
                theme_scores[theme] = score
            
            # Assign to theme with highest score
            if theme_scores:
                best_theme = max(theme_scores, key=theme_scores.get)
                if theme_scores[best_theme] > 0:
                    themes[best_theme].append(summary)
                else:
                    themes['conceptual'].append(summary)  # Default theme
        
        return themes
    
    def generate_abstract_overview(self, summaries: List[str]) -> str:
        """
        Generate a comprehensive abstract-style overview
        
        Args:
            summaries: List of chunk summaries
            
        Returns:
            Abstract overview text
        """
        logger.info("Generating abstract overview...")
        
        # Combine all summaries
        combined_text = ' '.join(summaries)
        
        # Extract key sentences (first sentence from each summary + important sentences)
        key_sentences = []
        
        for summary in summaries:
            sentences = [s.strip() for s in summary.split('.') if s.strip()]
            if sentences:
                # Always include first sentence
                key_sentences.append(sentences[0])
                
                # Include sentences with important keywords
                important_keywords = ['important', 'key', 'main', 'primary', 'essential', 
                                    'significant', 'crucial', 'fundamental', 'critical']
                
                for sentence in sentences[1:]:
                    if any(keyword in sentence.lower() for keyword in important_keywords):
                        key_sentences.append(sentence)
        
        # Organize into paragraphs
        paragraphs = []
        current_paragraph = []
        
        for i, sentence in enumerate(key_sentences):
            current_paragraph.append(sentence)
            
            # Create new paragraph every 3-4 sentences
            if len(current_paragraph) >= 3 and (len(current_paragraph) >= 4 or i == len(key_sentences) - 1):
                paragraph_text = '. '.join(current_paragraph) + '.'
                paragraphs.append(paragraph_text)
                current_paragraph = []
        
        # Add remaining sentences
        if current_paragraph:
            paragraph_text = '. '.join(current_paragraph) + '.'
            paragraphs.append(paragraph_text)
        
        # Ensure we have 3-5 paragraphs
        while len(paragraphs) < 3:
            # Split longer paragraphs
            longest_para = max(paragraphs, key=len)
            longest_idx = paragraphs.index(longest_para)
            
            sentences = longest_para.split('.')
            mid_point = len(sentences) // 2
            
            para1 = '. '.join(sentences[:mid_point]) + '.'
            para2 = '. '.join(sentences[mid_point:])
            
            paragraphs[longest_idx] = para1
            paragraphs.insert(longest_idx + 1, para2)
        
        # Limit to 5 paragraphs
        if len(paragraphs) > 5:
            paragraphs = paragraphs[:5]
        
        overview = '\n\n'.join(paragraphs)
        logger.info(f"Generated overview with {len(paragraphs)} paragraphs")
        
        return overview
    
    def extract_key_insights(self, summaries: List[str]) -> List[str]:
        """
        Extract 6-12 key insights from summaries
        
        Args:
            summaries: List of chunk summaries
            
        Returns:
            List of key insights
        """
        logger.info("Extracting key insights...")
        
        insights = []
        
        # Keywords that indicate important insights
        insight_indicators = [
            'important', 'key', 'significant', 'crucial', 'essential', 'fundamental',
            'main point', 'primary', 'critical', 'notable', 'remarkable', 'surprising',
            'interesting', 'valuable', 'useful', 'effective', 'successful', 'innovative'
        ]
        
        for summary in summaries:
            sentences = [s.strip() for s in summary.split('.') if s.strip()]
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                
                # Check for insight indicators
                if any(indicator in sentence_lower for indicator in insight_indicators):
                    # Clean and format the insight
                    insight = sentence.strip()
                    if insight and len(insight.split()) >= 5:  # Minimum length
                        insights.append(insight)
                
                # Also include sentences with numbers/statistics
                if re.search(r'\d+%|\d+x|\d+ times|\$\d+', sentence):
                    insight = sentence.strip()
                    if insight and len(insight.split()) >= 5:
                        insights.append(insight)
        
        # Remove duplicates while preserving order
        unique_insights = []
        seen = set()
        
        for insight in insights:
            # Create a simplified version for duplicate detection
            simplified = re.sub(r'[^\w\s]', '', insight.lower())
            if simplified not in seen:
                seen.add(simplified)
                unique_insights.append(insight)
        
        # Limit to 6-12 insights
        if len(unique_insights) > 12:
            unique_insights = unique_insights[:12]
        elif len(unique_insights) < 6:
            # Add more general insights from first sentences
            for summary in summaries:
                sentences = [s.strip() for s in summary.split('.') if s.strip()]
                if sentences and len(unique_insights) < 6:
                    first_sentence = sentences[0]
                    if first_sentence not in unique_insights:
                        unique_insights.append(first_sentence)
        
        logger.info(f"Extracted {len(unique_insights)} key insights")
        return unique_insights
    
    def generate_faqs(self, summaries: List[str]) -> List[Dict[str, str]]:
        """
        Generate FAQ pairs from content
        
        Args:
            summaries: List of chunk summaries
            
        Returns:
            List of FAQ dictionaries with 'question' and 'answer' keys
        """
        logger.info("Generating FAQs...")
        
        faqs = []
        
        # Common question patterns and their answers
        question_patterns = [
            (r'what is ([^.]+)', 'What is {}?'),
            (r'how to ([^.]+)', 'How do you {}?'),
            (r'why ([^.]+)', 'Why {}?'),
            (r'when ([^.]+)', 'When should you {}?'),
            (r'where ([^.]+)', 'Where can you {}?'),
            (r'([^.]+) works?', 'How does {} work?'),
            (r'benefits? of ([^.]+)', 'What are the benefits of {}?'),
            (r'advantages? of ([^.]+)', 'What are the advantages of {}?')
        ]
        
        # Extract potential Q&A pairs
        for summary in summaries:
            sentences = [s.strip() for s in summary.split('.') if s.strip()]
            
            for i, sentence in enumerate(sentences):
                sentence_lower = sentence.lower()
                
                # Look for explanatory sentences that could be answers
                if any(word in sentence_lower for word in ['because', 'since', 'due to', 'as a result']):
                    # Try to generate a question for this answer
                    if i > 0:
                        prev_sentence = sentences[i-1]
                        question = self._generate_question_from_context(prev_sentence, sentence)
                        if question:
                            faqs.append({
                                'question': question,
                                'answer': sentence
                            })
                
                # Look for definition-like sentences
                if ' is ' in sentence_lower or ' are ' in sentence_lower:
                    parts = sentence.split(' is ' if ' is ' in sentence else ' are ')
                    if len(parts) == 2:
                        subject = parts[0].strip()
                        definition = parts[1].strip()
                        
                        if len(subject.split()) <= 4:  # Keep questions concise
                            question = f"What is {subject}?"
                            faqs.append({
                                'question': question,
                                'answer': sentence
                            })
        
        # Add some general questions based on content themes
        themes = self.analyze_content_themes(summaries)
        
        general_questions = [
            "What are the main topics covered?",
            "What are the key takeaways?",
            "How can this information be applied?",
            "What are the most important points?",
            "What should viewers remember most?"
        ]
        
        # Generate answers for general questions
        for question in general_questions:
            if len(faqs) < 8:  # Limit total FAQs
                answer = self._generate_general_answer(question, summaries)
                if answer:
                    faqs.append({
                        'question': question,
                        'answer': answer
                    })
        
        # Limit to 5-8 FAQs
        if len(faqs) > 8:
            faqs = faqs[:8]
        
        logger.info(f"Generated {len(faqs)} FAQ pairs")
        return faqs
    
    def _generate_question_from_context(self, context: str, answer: str) -> Optional[str]:
        """Generate a question based on context and answer"""
        context_lower = context.lower()
        
        if 'what' in context_lower:
            return f"What {context.split('what')[1].strip()}?"
        elif 'how' in context_lower:
            return f"How {context.split('how')[1].strip()}?"
        elif 'why' in context_lower:
            return f"Why {context.split('why')[1].strip()}?"
        
        return None
    
    def _generate_general_answer(self, question: str, summaries: List[str]) -> Optional[str]:
        """Generate answer for general questions"""
        question_lower = question.lower()
        
        if 'main topics' in question_lower or 'covered' in question_lower:
            # Extract main topics from summaries
            topics = []
            for summary in summaries[:3]:  # Use first few summaries
                sentences = summary.split('.')
                if sentences:
                    topics.append(sentences[0].strip())
            return f"The main topics include: {', '.join(topics[:3])}."
        
        elif 'key takeaways' in question_lower or 'important points' in question_lower:
            # Use first sentence from each summary
            takeaways = []
            for summary in summaries[:3]:
                sentences = summary.split('.')
                if sentences:
                    takeaways.append(sentences[0].strip())
            return f"Key takeaways include: {'. '.join(takeaways)}."
        
        elif 'applied' in question_lower:
            # Look for practical applications
            practical_content = []
            for summary in summaries:
                if any(word in summary.lower() for word in ['use', 'apply', 'implement', 'practice']):
                    sentences = summary.split('.')
                    practical_content.extend(sentences[:2])
            
            if practical_content:
                return f"This information can be applied by: {'. '.join(practical_content[:2])}."
        
        return None
    
    def identify_hidden_gems(self, summaries: List[str]) -> List[str]:
        """
        Identify hidden gems and actionable tips
        
        Args:
            summaries: List of chunk summaries
            
        Returns:
            List of hidden gems/tips
        """
        logger.info("Identifying hidden gems and actionable tips...")
        
        gems = []
        
        # Keywords that indicate actionable content
        actionable_keywords = [
            'tip', 'trick', 'hack', 'secret', 'technique', 'method', 'strategy',
            'approach', 'way to', 'how to', 'you can', 'try', 'consider',
            'remember', 'keep in mind', 'important to', 'make sure'
        ]
        
        # Keywords that indicate valuable insights
        insight_keywords = [
            'surprising', 'unexpected', 'interesting', 'remarkable', 'notable',
            'little known', 'hidden', 'secret', 'insider', 'expert',
            'advanced', 'pro tip', 'bonus', 'extra'
        ]
        
        for summary in summaries:
            sentences = [s.strip() for s in summary.split('.') if s.strip()]
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                
                # Check for actionable content
                if any(keyword in sentence_lower for keyword in actionable_keywords):
                    if len(sentence.split()) >= 8:  # Ensure substantial content
                        gems.append(sentence)
                
                # Check for valuable insights
                elif any(keyword in sentence_lower for keyword in insight_keywords):
                    if len(sentence.split()) >= 6:
                        gems.append(sentence)
                
                # Check for specific patterns
                elif re.search(r'(always|never|avoid|ensure|remember to)', sentence_lower):
                    if len(sentence.split()) >= 6:
                        gems.append(sentence)
        
        # Remove duplicates and limit to 3-5 gems
        unique_gems = []
        seen = set()
        
        for gem in gems:
            simplified = re.sub(r'[^\w\s]', '', gem.lower())
            if simplified not in seen and len(unique_gems) < 5:
                seen.add(simplified)
                unique_gems.append(gem)
        
        # If we don't have enough gems, add some general tips
        if len(unique_gems) < 3:
            for summary in summaries:
                sentences = summary.split('.')
                for sentence in sentences:
                    if ('important' in sentence.lower() or 'key' in sentence.lower()) and len(unique_gems) < 3:
                        if sentence.strip() not in unique_gems:
                            unique_gems.append(sentence.strip())
        
        logger.info(f"Identified {len(unique_gems)} hidden gems")
        return unique_gems
    
    def build_final_summary(self, summaries: List[str], save_to_file: bool = True, 
                           generate_pdf: bool = False) -> Dict[str, Any]:
        """
        Build the final comprehensive summary
        
        Args:
            summaries: List of chunk summaries
            save_to_file: Whether to save to text file
            generate_pdf: Whether to generate PDF
            
        Returns:
            Dictionary containing all summary sections
        """
        logger.info("Building final comprehensive summary...")
        
        # Generate all sections
        abstract = self.generate_abstract_overview(summaries)
        insights = self.extract_key_insights(summaries)
        faqs = self.generate_faqs(summaries)
        gems = self.identify_hidden_gems(summaries)
        
        # Build final summary structure
        final_summary = {
            'abstract': abstract,
            'insights': insights,
            'faqs': faqs,
            'gems': gems,
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'chunk_count': len(summaries),
                'total_sections': 4
            }
        }
        
        if save_to_file:
            self._save_text_summary(final_summary)
        
        if generate_pdf and REPORTLAB_AVAILABLE:
            self._save_pdf_summary(final_summary)
        elif generate_pdf:
            logger.warning("PDF generation requested but reportlab not available")
        
        logger.info("Final summary completed successfully")
        return final_summary
    
    def _save_text_summary(self, summary: Dict[str, Any]):
        """Save summary to text file"""
        output_path = Path("youtube_summarizer/outputs/final/final_summary.txt")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("YOUTUBE VIDEO ANALYSIS - COMPREHENSIVE SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            # Abstract Overview
            f.write("ABSTRACT OVERVIEW\n")
            f.write("-" * 20 + "\n\n")
            f.write(summary['abstract'] + "\n\n")
            
            # Key Insights
            f.write("KEY INSIGHTS\n")
            f.write("-" * 15 + "\n\n")
            for i, insight in enumerate(summary['insights'], 1):
                f.write(f"{i}. {insight}\n")
            f.write("\n")
            
            # FAQs
            f.write("FREQUENTLY ASKED QUESTIONS\n")
            f.write("-" * 30 + "\n\n")
            for faq in summary['faqs']:
                f.write(f"Q: {faq['question']}\n")
                f.write(f"A: {faq['answer']}\n\n")
            
            # Hidden Gems
            f.write("HIDDEN GEMS & ACTIONABLE TIPS\n")
            f.write("-" * 35 + "\n\n")
            for i, gem in enumerate(summary['gems'], 1):
                f.write(f"{i}. {gem}\n")
            
            # Metadata
            f.write(f"\n\n" + "=" * 60 + "\n")
            f.write(f"Generated: {summary['metadata']['generated_at']}\n")
            f.write(f"Processed {summary['metadata']['chunk_count']} content chunks\n")
        
        logger.info(f"Text summary saved to: {output_path}")
    
    def _save_pdf_summary(self, summary: Dict[str, Any]):
        """Save summary to PDF file"""
        if not REPORTLAB_AVAILABLE:
            return
        
        output_path = Path("youtube_summarizer/outputs/final/final_summary.pdf")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        doc = SimpleDocTemplate(str(output_path), pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=20
        )
        
        story = []
        
        # Title
        story.append(Paragraph("YouTube Video Analysis", title_style))
        story.append(Paragraph("Comprehensive Summary", title_style))
        story.append(Spacer(1, 20))
        
        # Abstract Overview
        story.append(Paragraph("Abstract Overview", heading_style))
        story.append(Paragraph(summary['abstract'], styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Key Insights
        story.append(Paragraph("Key Insights", heading_style))
        for i, insight in enumerate(summary['insights'], 1):
            story.append(Paragraph(f"{i}. {insight}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # FAQs
        story.append(Paragraph("Frequently Asked Questions", heading_style))
        for faq in summary['faqs']:
            story.append(Paragraph(f"<b>Q:</b> {faq['question']}", styles['Normal']))
            story.append(Paragraph(f"<b>A:</b> {faq['answer']}", styles['Normal']))
            story.append(Spacer(1, 10))
        
        # Hidden Gems
        story.append(Paragraph("Hidden Gems & Actionable Tips", heading_style))
        for i, gem in enumerate(summary['gems'], 1):
            story.append(Paragraph(f"{i}. {gem}", styles['Normal']))
        
        doc.build(story)
        logger.info(f"PDF summary saved to: {output_path}")

# Main function for external use
def build_final_summary(summaries: List[str], save_to_file: bool = True, 
                       generate_pdf: bool = False) -> Dict[str, Any]:
    """
    Main function to build final summary from chunk summaries
    
    Args:
        summaries: List of chunk summaries
        save_to_file: Whether to save to text file
        generate_pdf: Whether to generate PDF
        
    Returns:
        Dictionary containing final summary
    """
    aggregator = SummaryAggregator()
    return aggregator.build_final_summary(summaries, save_to_file, generate_pdf)

if __name__ == "__main__":
    # Test the aggregator
    sample_summaries = [
        "Artificial intelligence is transforming various industries through machine learning algorithms. These systems can process large amounts of data to identify patterns and make predictions.",
        "Deep learning networks use multiple layers to analyze complex data structures. This approach has been particularly successful in image recognition and natural language processing tasks.",
        "The implementation of AI systems requires careful consideration of ethical implications. Issues such as bias, privacy, and transparency must be addressed to ensure responsible deployment.",
        "Practical applications of AI include recommendation systems, autonomous vehicles, and medical diagnosis tools. These technologies are becoming increasingly sophisticated and reliable."
    ]
    
    print("Testing summary aggregator...")
    
    try:
        final_summary = build_final_summary(sample_summaries, save_to_file=False)
        
        print("\n=== FINAL SUMMARY ===")
        print(f"\nAbstract ({len(final_summary['abstract'].split())} words):")
        print(final_summary['abstract'])
        
        print(f"\nKey Insights ({len(final_summary['insights'])} items):")
        for i, insight in enumerate(final_summary['insights'], 1):
            print(f"  {i}. {insight}")
        
        print(f"\nFAQs ({len(final_summary['faqs'])} items):")
        for faq in final_summary['faqs']:
            print(f"  Q: {faq['question']}")
            print(f"  A: {faq['answer']}")
        
        print(f"\nHidden Gems ({len(final_summary['gems'])} items):")
        for i, gem in enumerate(final_summary['gems'], 1):
            print(f"  {i}. {gem}")
        
        print("\n✅ Aggregator test completed successfully")
        
    except Exception as e:
        print(f"❌ Error: {e}")
