"""
Computer Vision Analysis Module
Detects fraudulent patterns in images and documents
"""

import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import re
from typing import Dict, Any, Optional, List
import urllib.parse

class VisionAnalyzer:
    """
    Computer Vision module for job posting fraud detection
    """
    
    def __init__(self):
        self.suspicious_url_patterns = [
            r'bit\.ly',
            r'tinyurl',
            r'short\.link',
            r'goo\.gl'
        ]
    
    def validate_company_website(self, website_url: str) -> Dict[str, Any]:
        """
        Validate company website for authenticity
        """
        try:
            if not website_url or not website_url.strip():
                return {
                    'is_valid': False,
                    'has_https': False,
                    'suspicious_url': False,
                    'domain_age': 'Unknown',
                    'website_reachable': False,
                    'verdict': 'No website provided'
                }
            
            # Check URL format
            has_https = website_url.startswith('https://')
            
            # Check for suspicious URL shorteners
            suspicious_url = any(re.search(pattern, website_url, re.IGNORECASE) 
                               for pattern in self.suspicious_url_patterns)
            
            # Try to reach website
            website_reachable = self._check_website_reachable(website_url)
            
            # Suspicious patterns
            verdict_factors = []
            if not has_https:
                verdict_factors.append('No HTTPS protection')
            if suspicious_url:
                verdict_factors.append('Suspicious URL shortener detected')
            if not website_reachable:
                verdict_factors.append('Website not reachable')
            
            verdict = 'SUSPICIOUS' if verdict_factors else 'VALID'
            
            return {
                'is_valid': website_reachable and has_https,
                'has_https': has_https,
                'suspicious_url': suspicious_url,
                'website_reachable': website_reachable,
                'verdict': verdict,
                'red_flags': verdict_factors
            }
        
        except Exception as e:
            return {
                'is_valid': False,
                'error': str(e),
                'verdict': 'ERROR'
            }
    
    def detect_logo_spoofing(self, image_url: Optional[str] = None, 
                            company_name: str = '') -> Dict[str, Any]:
        """
        Detect if company logo appears to be spoofed or generic
        """
        try:
            if not image_url:
                return {
                    'has_image': False,
                    'is_spoofed': 'Unknown',
                    'confidence': 0,
                    'message': 'No image provided'
                }
            
            # Try to download and analyze image
            image = self._download_image(image_url)
            
            if image is None:
                return {
                    'has_image': False,
                    'is_spoofed': 'Unknown',
                    'confidence': 0,
                    'message': 'Could not download image'
                }
            
            # Analyze image characteristics
            features = self._analyze_image_features(image)
            
            # Check for generic/stock image indicators
            is_generic = (
                features['color_diversity'] < 0.3 or
                features['has_watermark'] or
                features['histogram_peak_height'] > 0.5
            )
            
            return {
                'has_image': True,
                'is_spoofed': is_generic,
                'confidence': 0.7 if is_generic else 0.3,
                'features': features,
                'message': 'Logo appears generic/stock' if is_generic else 'Logo appears original'
            }
        
        except Exception as e:
            return {
                'has_image': False,
                'error': str(e),
                'message': f'Error analyzing image: {str(e)}'
            }
    
    def extract_text_from_image(self, image_url: str) -> str:
        """
        Extract text from image using OCR
        Requires pytesseract and tesseract installation
        """
        try:
            image = self._download_image(image_url)
            if image is None:
                return ""
            
            # Try to use pytesseract if available
            try:
                import pytesseract
                text = pytesseract.image_to_string(image)
                return text
            except ImportError:
                # Fallback: return empty if pytesseract not installed
                return ""
        
        except Exception as e:
            return ""
    
    def verify_company_information(self, company_name: str, 
                                  location: str) -> Dict[str, Any]:
        """
        Cross-verify company information for authenticity
        """
        verification_results = {
            'company_name': company_name,
            'location': location,
            'is_suspicious': False,
            'flags': []
        }
        
        # Check for generic company names
        generic_names = ['company', 'business', 'work', 'job', 'corporation', 'corp', 'inc', 'limited']
        if any(generic in company_name.lower() for generic in generic_names):
            if company_name.lower() in generic_names or len(company_name.split()) == 1:
                verification_results['flags'].append('Very generic company name')
                verification_results['is_suspicious'] = True
        
        # Check for suspicious location patterns
        suspicious_locations = ['remote worldwide', 'any location', 'worldwide']
        if any(sus_loc in location.lower() for sus_loc in suspicious_locations):
            verification_results['flags'].append('Unusually broad location')
        
        # Check name format (legitimate companies usually have proper format)
        if not self._is_proper_name_format(company_name):
            verification_results['flags'].append('Unusual company name format')
        
        return verification_results
    
    def detect_qr_codes(self, image_url: str) -> Dict[str, Any]:
        """
        Detect QR codes in images (often used to hide malicious links)
        """
        try:
            image = self._download_image(image_url)
            if image is None:
                return {'has_qr_code': False, 'message': 'Could not load image'}
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Try to detect QR code patterns using edge detection
            edges = cv2.Canny(gray, 100, 200)
            
            # Check for typical QR code features
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            has_qr = len(contours) > 5  # Simplified check
            
            return {
                'has_qr_code': has_qr,
                'warning': 'QR code detected - verify link safety' if has_qr else 'No QR code detected'
            }
        
        except Exception as e:
            return {
                'has_qr_code': False,
                'error': str(e)
            }
    
    # Helper Methods
    
    def _download_image(self, url: str) -> Optional[np.ndarray]:
        """Download image from URL"""
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                image_data = Image.open(BytesIO(response.content))
                return cv2.cvtColor(np.array(image_data), cv2.COLOR_RGB2BGR)
        except (requests.RequestException, IOError):
            pass
        return None
    
    def _analyze_image_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze image characteristics"""
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate color diversity
        unique_colors = len(np.unique(hsv.reshape(-1, 3), axis=0))
        total_colors = hsv.shape[0] * hsv.shape[1]
        color_diversity = unique_colors / total_colors
        
        # Check histogram
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        hist_peak = np.max(hist)
        hist_peak_height = hist_peak / np.sum(hist) if np.sum(hist) > 0 else 0
        
        # Simple watermark detection (check for text-like patterns)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        has_watermark = np.sum(edges) > (edges.shape[0] * edges.shape[1] * 0.1)
        
        return {
            'color_diversity': color_diversity,
            'histogram_peak_height': hist_peak_height,
            'has_watermark': has_watermark,
            'image_shape': image.shape
        }
    
    def _check_website_reachable(self, url: str) -> bool:
        """Check if website is reachable"""
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            response = requests.head(url, timeout=5, allow_redirects=True)
            return response.status_code < 400
        except requests.RequestException:
            return False
    
    def _is_proper_name_format(self, name: str) -> bool:
        """Check if name follows proper formatting"""
        # Check if name has proper capitalization and is not all caps
        has_lowercase = any(c.islower() for c in name)
        has_uppercase = any(c.isupper() for c in name)
        is_not_all_caps = not name.isupper()
        
        return has_lowercase and has_uppercase and is_not_all_caps


if __name__ == "__main__":
    analyzer = VisionAnalyzer()
    
    # Test company website validation
    result = analyzer.validate_company_website("https://google.com")
    print("Website Validation:", result)
    
    # Test company information verification
    result = analyzer.verify_company_information("Company Corp", "Remote")
    print("Company Verification:", result)
