#!/usr/bin/env python3
import io
import json
import base64

from PIL import Image
import numpy as np

from nicegui import run, ui
from nicegui.events import UploadEventArguments, MouseEventArguments

modelcard_path = "cfg/models.json"


@ui.refreshable
def get_modelcards():
    with open(modelcard_path) as f:
        return json.load(f)

def generate_card_image(image_path, mask_path):
    # Open the image file
    with Image.open(image_path) as img:
        # Open the mask file
        with Image.open(mask_path) as mask:
            # Ensure the mask is in RGBA mode
            mask = mask.convert('RGBA')
            # Resize the mask to match the image size
            mask = mask.resize(img.size)
            # Combine the image and mask
            combined = Image.alpha_composite(img.convert('RGBA'), mask)
            combined.save('combined_image.png')

def model_card(card):
    preview = preview_modelcard(card)
    with ui.card().style('position: relative; width: 400px; height: 225px; padding: 10px; border-width: 1px; solid: #eee; border-radius: 15px').tight():
        # with ui.element('div').style('position: relative; width: 100%; height: 30%; overflow: hidden;'):
            # if 'card_image' in card:
            #     ui.image(card['card_image']).style('position: static; width: 400px; height: 100px; object-fit: none;')
        ui.label(card['model_name']).classes('break-word text-2xl').style('position: static; text-shadow: 2px 2px #ffffff;')
        with ui.row().style('gap: 10px; padding-bottom: 10px'):
            for model_type in card['model_type']:
                ui.label(model_type).style('font-size: 12px; padding-bottom: 10px; color: #888') 
        ui.label(card['description']).style('color: #666; display: -webkit-box; -webkit-box-orient: vertical; -webkit-line-clamp: 4; overflow: hidden;').classes('break-word text-1xl')
        with ui.row().style('position: absolute; bottom: 10px; width: 15em; gap: 10px; align-items: center;'):
            ui.button('preview', on_click=preview.open).style('width: 100px')
            ui.button('Enter', on_click=lambda: ui.navigate.to(f'/ModelBuilder/{card["model_name"]}')).style('width: 100px')





def get_new_model_popup(on_create_callback: callable):
    with ui.dialog().props('backdrop-filter="blur(8px) brightness(80%)"') as new_model_popup, ui.card().style('max-width: 2000px; width: 500px; border-radius: 15px;'):
        modelcard_cfg = {"model_name": "",
                         "description": "",
                         "model_type": ""
                        }
        def update_modelcard_cfg(comp, value):
            if comp == "model_type":
                modelcard_cfg[comp] = [value]
            else:
                modelcard_cfg[comp] = value

        ui.label('Create a New Model').classes('text-3xl')
        ui.input(label='Model Name', on_change=lambda e: update_modelcard_cfg("model_name", e.value)).classes('w-full')
        ui.textarea(label='Description', on_change=lambda e: update_modelcard_cfg("description", e.value)).classes('w-full')
        ui.input(label='Model Type (e.g., LLM, Vision, Audio)', on_change=lambda e: update_modelcard_cfg("model_type", e.value)).classes('w-full')
        
        def handle_create():
            save_new_modelcard(modelcard_cfg)
            new_model_popup.close()
            on_create_callback() # This is the key: trigger the refresh
        ui.button('Create', on_click=handle_create)
        
    return new_model_popup


def save_new_modelcard(cfg):
    print(cfg)
    with open(modelcard_path, 'r+') as f:
        modelcards = json.load(f)
        modelcards.append(cfg)
        f.seek(0)
        json.dump(modelcards, f, indent=4)
        f.truncate()


def preview_modelcard(card):
    with ui.dialog().props('backdrop-filter="blur(8px) brightness(80%)"') as modelcard_preview, ui.card().style('max-width: 2000px; width: 500px; border-radius: 15px;'):
        # with ui.row().style('padding: 10px'):
        #     ui.image(card['card_image']).style('width: 100%; height: auto; border-radius: 10px;')
        with ui.column().style('padding: 10px; gap: 10px;').classes('w-full'):
            ui.label(card['model_name']).classes('text-3xl')
            ui.label(", ".join(card['model_type']))
        with ui.column().style('padding: 10px'):
            ui.label('')
            ui.label("Description:").classes('text-xl')
            ui.label(card['description'])

    return modelcard_preview

# Home Page
@ui.page('/')
def home_page():

    def build_model_cards_display():
        model_cards_row.clear()  # Important: remove old cards
        with model_cards_row:
            model_cards = get_modelcards()
            for card in model_cards:
                model_card(card)
    new_model_popup = get_new_model_popup(build_model_cards_display)
    with ui.row().style('padding-left: 20px; padding-top: 20px; padding-bottom: 40px'):
        ui.label("Home").classes('text-4xl')
    with ui.row().style('padding-left: 20px; padding-bottom: 20px; gap: 50px'):
        ui.label("My Models").classes('text-3xl')
        ui.button('+ New Model', on_click=new_model_popup.open)
    with ui.row().style('padding-left: 20px; padding-bottom: 20px; gap: 100px') as model_cards_row:
        model_cards = get_modelcards()
        for card in model_cards:
            model_card(card)


@ui.page('/ModelBuilder/{model_name}')
def model_builder_page(model_name: str):
    all_cards = get_modelcards()
    card = next((c for c in all_cards if c['model_name'] == model_name), None)
    if card is None:
        ui.label("Model not found!")
        return

    module_list_json = [
        {
            "name": "Attention"
        },
        {
            "name": "CNN"
        },
        {
            "name": "RNN"
        },
        {
            "name": "CNN"
        },
        {
            "name": "RNN"
        },
        {
            "name": "CNN"
        },
        {
            "name": "RNN"
        }
    ]

    function_list_json = [
        {
            "name": "Relu"
        },
        {
            "name": "Softmax"
        },
        {
            "name": "Tanh"
        }
    ]

    def module_list_component():
        with ui.column().style('position: fixed; left: 100px; top: 20%; gap: 8px; max-height: 40vh; padding: 10px;') as module_list:
            ui.label("Modules").classes('text-2xl').style('margin-bottom: 10px;')
            with ui.column().style('overflow-y: auto; max-height: calc(40vh - 80px);'):
                for module in module_list_json:
                    ui.button(module['name'], on_click=lambda m=module: ui.notify(f"Selected module: {m['name']}")).classes('text-md').style('width: 180px; height: 30px; border-radius: 10px; background-color: #f0f0f0; color: #333; margin-bottom: 5px; flex-shrink: 0;').props('outline="none"')
        return module_list
    
    def function_list_component():
        with ui.column().style('position: fixed; left: 100px; top: 20%; gap: 8px; max-height: 40vh; padding: 10px;') as function_list:
            ui.label("Functions").classes('text-2xl').style('margin-bottom: 10px;')
            with ui.column().style('overflow-y: auto; max-height: calc(40vh - 80px);'):
                for function in function_list_json:
                    ui.button(function['name'], on_click=lambda f=function: ui.notify(f"Selected function: {f['name']}")).classes('text-md').style('width: 180px; height: 30px; border-radius: 10px; background-color: #f0f0f0; color: #333; margin-bottom: 5px; flex-shrink: 0;').props('outline="none"')
        return function_list


    def handle_module_list():
        if function_list.visible:
            function_list.visible = False
        module_list.visible = not module_list.visible

    def handle_function_list():
        if module_list.visible:
            module_list.visible = False
        function_list.visible = not function_list.visible

    
    def handle_module_click(module):
        module_name = module['name']


    with ui.row().style('padding-left: 20px; padding-top: 20px; padding-bottom: 20px; align-items: center; gap: 20px;'):
        ui.image("webui/assets/back.svg").style('width: 30px; height: 30px; cursor: pointer;') \
            .on('click', lambda: ui.navigate.to('/'))
        ui.label(f"Model Builder/{model_name}").classes('text-4xl')
    with ui.column().style('position: fixed; left: 20px; top: 25%; transform: translateY(-50%);'):
        ui.button("M", on_click=handle_module_list).style('width: 50px; height: 50px; border-radius: 25px;').props('outline="none"')
        ui.button("F", on_click=handle_function_list).style('width: 50px; height: 50px; border-radius: 25px;').props('outline="none"')

    module_list = module_list_component()
    module_list.visible = False
    function_list = function_list_component()
    function_list.visible = False
    

def main():
    home_page()
    ui.run(port=8000, title='LLMBase - Build your own LLMs with ease')


main()