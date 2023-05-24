from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import FieldList, FormField, DateField, FloatField, StringField, SelectField
from wtforms.validators import DataRequired
from wtforms import Form as NoCsrfForm


class Exercise(NoCsrfForm):
    expense_name = StringField('Expense_Item', validators=[DataRequired()])
    cost = FloatField('Cost', validators=[DataRequired()])
    due_date = DateField('Due Date', format='%Y-%m-%d',
                                 validators=[DataRequired()])
    type = SelectField('Role', choices=[
        ('mutual', 'Mutual'),
        ('personal#1', 'Personal #1'),
        ('personal#2', 'Personal #2'),
    ])

class Workout(FlaskForm):
    """A collection of exercises."""
    items = FieldList(FormField(Exercise), min_entries=1)