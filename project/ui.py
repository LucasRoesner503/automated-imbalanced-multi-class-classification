import FreeSimpleGUI as sg
from ml import execute_byCharacteristics

sg.theme('Dark Blue 3')

layout = [
    [sg.Text('Automated Imbalanced Classification')],
    [sg.Text('Dataset file location ', size=(15, 1)), sg.InputText(key="file"), sg.FileBrowse()],
    [sg.Text('OpenML dataset ID ', size=(15, 1)), sg.InputText(key="omid")],
    [sg.Text('Problem type ', size=(15, 1)),
     sg.Combo(['Auto-detect', 'Binary', 'Multiclass'], default_value='Auto-detect', key='problem_type', readonly=True)],
    [sg.Submit(), sg.Cancel()],
]

window = sg.Window('Automated Imbalanced Classification', layout)

while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED or event == 'Cancel':
        break

    elif values['file'] and values['omid']:
        sg.Popup("Please only choose a dataset file or an OpenML dataset ID, not both!\n", keep_on_top=True, title='Error')

    elif values['file'] or values['omid']:
        selection = values['problem_type']
        problem_type = None if selection == 'Auto-detect' else selection.lower()

        if values['file']:
            str_output = execute_byCharacteristics(values['file'], None, problem_type=problem_type)
        else:
            str_output = execute_byCharacteristics(None, values['omid'], problem_type=problem_type)

        sg.Popup(str_output, keep_on_top=True, title='Recommendations')

    else:
        sg.Popup("Please choose a dataset file or put an OpenML dataset ID!\n", keep_on_top=True, title='Error')

window.close()
