from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, TextAreaField, SelectField
from wtforms.validators import DataRequired, Length

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=20)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    submit = SubmitField('Login')

class RegisterForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=20)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    preferences = TextAreaField('Preferences (optional)')
    submit = SubmitField('Register')

class CheckoutForm(FlaskForm):
    submit = SubmitField('Proceed to Checkout')

class PaymentForm(FlaskForm):
    crypto = SelectField('Cryptocurrency', choices=[
        ('PI', 'PI'), ('BTC', 'BTC'), ('ETH', 'ETH'),
        ('USDT', 'USDT'), ('USDC', 'USDC'), ('SOL', 'SOL')
    ], validators=[DataRequired()])
    submit = SubmitField('Pay Now')