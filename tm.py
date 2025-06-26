from nicegui import ui

# A list of interests to be displayed in the checklist
INTERESTS = ['Reading', 'Gaming', 'Hiking', 'Cooking', 'Coding', 'Sports']

# A container to hold the results, instead of a single label
with ui.card().classes('w-full'):
    ui.label('Result from the dialog will be shown here.').classes('text-gray-500')
    result_area = ui.column()


async def show_input_dialog():
    """
    This function defines and handles the input dialog, now with a textarea.
    """
    # The card's height will adjust to the content automatically.
    with ui.dialog().props('backdrop-filter="blur(8px)"') as dialog, \
         ui.card().style('max-width: 1000px; width: 1000px; border-radius: 15px;') as card:
        
        ui.label('Please enter your details:').classes('text-h6')
        
        name_input = ui.input(label='Name')
        age_input = ui.number(label='Age', format='%.0f')
        
        # Add the textarea for multi-line introduction
        introduction_textarea = ui.textarea(label='Introduction', 
                                            placeholder='Tell us a bit about yourself...')

        interests_checklist = ui.checkbox(INTERESTS)
        
        ui.space()

        with ui.row().classes('w-full justify-end'):
            # Submit button now also includes the textarea's value
            ui.button('Submit', on_click=lambda: dialog.submit({
                'name': name_input.value,
                'age': age_input.value,
                'introduction': introduction_textarea.value,
                'interests': interests_checklist.value,
            }))

    result = await dialog

    # Clear the previous results and display the new ones
    result_area.clear()
    if result:
        # Using a `with` statement on the result_area container to add new elements
        with result_area:
            interests_str = ', '.join(result['interests']) if result['interests'] else 'None'
            ui.label(f"Name: {result['name']}").classes('text-lg font-bold')
            ui.label(f"Age: {result['age']}")
            ui.label(f"Interests: {interests_str}")
            ui.label("Introduction:").classes('mt-4 font-bold')
            # Using ui.markdown is great for safely displaying user-provided multi-line text
            ui.markdown(result['introduction'] or 'No introduction provided.')
    else:
        with result_area:
            ui.label('You closed the dialog without submitting.')


# The button on the main page that triggers the dialog
ui.button('Open Input Dialog', on_click=show_input_dialog)

ui.run()