from flask import Flask, render_template, request
import requests

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    weather_data = None
    error_msg = None
    
    if request.method == 'POST':
        city = request.form.get('city')
        if city:
            api_key = "b2a757657153a7741d40807b1c3e3871" 
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
            
            try:
                response = requests.get(url).json()
                
                if response.get('cod') == 200:
                    weather_data = {
                        'city': response['name'],
                        'country': response['sys']['country'],
                        'temp': response['main']['temp'],
                        'description': response['weather'][0]['description'].capitalize(),
                        'humidity': response['main']['humidity'],
                        'wind_speed': response['wind']['speed'],
                        'icon': response['weather'][0]['icon']
                    }
                else:
                    error_msg = "invalid choice."
            except Exception as e:
                error_msg = "API connection failed."
                
    return render_template('index.html', weather=weather_data, error=error_msg)

if __name__ == '__main__':
    app.run(debug=True)
