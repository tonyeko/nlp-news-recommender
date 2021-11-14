# set FLASK_APP=Web.py
# 

from flask import Flask, render_template
from flask import request, redirect, url_for
from Algoritma import mainProgram


app = Flask(__name__)


@app.route("/", methods=["POST", "GET"])
def home():
    if request.method == "POST":
        return redirect(url_for('form'))
    else:
        return render_template('home.html')


@app.route("/form", methods=["POST", "GET"])
def form():
    if request.method == 'POST':
        if request.form['submit'] == 'oke':
            foldr = request.form["folder"]
            if foldr != '':
                hasil = mainProgram(foldr)
                return render_template('form.html', error=hasil)
            else:
                return render_template('form.html')
        elif request.form['submit'] == 'home':
            return redirect(url_for('home'))
        else:
            return redirect(url_for('about'))
    else:
        return render_template('form.html')


@app.route("/about", methods=["POST", "GET"])
def about():
    if request.method == "POST":
        if request.form['submit'] == 'home':
            return redirect(url_for('home'))
    else:
        return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)
