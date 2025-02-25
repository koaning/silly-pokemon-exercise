# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "groq==0.18.0",
#     "instructor==1.7.2",
#     "lazylines==0.0.4",
#     "marimo",
#     "pandas==2.2.3",
#     "polars==1.23.0",
#     "pydantic==2.10.6",
#     "python-dotenv==1.0.1",
#     "tqdm==4.67.1",
# ]
# ///

import marimo

__generated_with = "0.11.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import pandas as pd
    return mo, pd, pl


@app.cell
def _():
    import os
    from groq import Groq
    import instructor
    from pydantic import BaseModel
    from dotenv import load_dotenv 

    load_dotenv(".env")
    return BaseModel, Groq, instructor, load_dotenv, os


@app.cell
def _(os):
    os.getenv("GROQ_API_KEY")
    return


@app.cell
def _(BaseModel, Groq, instructor, os):


    # Enable instructor patches for Groq client
    client = instructor.from_groq(Groq(api_key=os.getenv("GROQ_API_KEY")))


    class PokemonDesc(BaseModel):
        name: str
        description: str


    def get_pokemon_desc(name):
        return client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": f"Describe the following pokemon by using 2-3 sentences by describing the appearance, character and moves: {name}"},
            ],
            response_model=PokemonDesc,
        ).description
    return PokemonDesc, client, get_pokemon_desc


@app.cell
def _(get_pokemon_desc):
    get_pokemon_desc("pikachu")
    return


@app.cell
def _(pl):
    pokemons = pl.read_ndjson("data/pokemon.jsonl").select("name", "type1").to_dicts()
    return (pokemons,)


@app.cell
def _(get_pokemon_desc):
    get_pokemon_desc("raichu")
    return


@app.cell
def _(get_pokemon_desc, mo, pokemons):
    from tqdm import tqdm 
    from lazylines import LazyLines

    data = [] 
    for poke in mo.status.progress_bar(pokemons):
        poke["desc"] = get_pokemon_desc(poke['name'])
        data.append(poke)
    return LazyLines, data, poke, tqdm


@app.cell
def _(data, pl):
    pl.DataFrame(data).write_ndjson("data/pokemon_desc.jsonl")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
