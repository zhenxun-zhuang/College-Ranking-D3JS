from flask import Flask, render_template

app = Flask(__name__)

@app.route("/", methods=['GET','POST'])
def index():
    return render_template("index.html")

@app.route("/annual", methods=['GET','POST'])
def annual():
    return render_template("annual.html")
	
@app.route("/changes", methods=['GET','POST'])
def changes():
    return render_template("changes.html")
	
@app.route("/countries", methods=['GET','POST'])
def countries():
    return render_template("countries.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8888,debug=True)