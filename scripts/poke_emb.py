# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair==5.5.0",
#     "marimo",
#     "matplotlib==3.10.0",
#     "numpy==2.1.3",
#     "polars==1.23.0",
#     "scikit-learn==1.6.1",
#     "sentence-transformers==3.4.1",
#     "umap-learn==0.5.7",
# ]
# ///

import marimo

__generated_with = "0.11.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import altair as alt
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer, alt, mo, np, pl


@app.cell
def _(SentenceTransformer, pl):
    descs = pl.read_ndjson("data/pokemon_desc.jsonl")['desc'].to_list()
    embs = SentenceTransformer("all-MiniLM-L6-v2").encode(descs)
    return descs, embs


@app.cell
def _(embs):
    from umap import UMAP 
    from sklearn.decomposition import PCA 

    X_tfm = UMAP().fit_transform(embs)
    # X_tfm = PCA().fit_transform(embs)
    return PCA, UMAP, X_tfm


@app.cell
def _(pl):
    poke_types = pl.read_ndjson("data/pokemon_desc.jsonl")['type1'].to_list()
    return (poke_types,)


@app.cell
def _(X_tfm, pl, poke_types):
    embedding = pl.DataFrame(
        {"x": X_tfm[:, 0], "y": X_tfm[:, 1], "type": poke_types, "index": range(len(poke_types))}
    ).filter(pl.col("index") < 700)
    return (embedding,)


@app.cell(hide_code=True)
def _(embedding, mo, scatter):
    chart = mo.ui.altair_chart(scatter(embedding))
    chart
    return (chart,)


@app.cell(hide_code=True)
def _(chart, images, mo, np):
    mo.stop(not len(chart.value))

    def show_images(indices, max_images=10):
        import matplotlib.pyplot as plt
        from PIL import Image

        img_paths = [f"data/pokemon_jpg/{_}.jpg" for _ in chart.value['index']][:max_images]
        fig, axes = plt.subplots(1, len(img_paths))
        fig.set_size_inches(12.5, 1.5)
        if len(indices) > 1:
            for im, ax in zip(img_paths, axes.flat):
                img = Image.open(im)
                img_array = np.array(img)
                ax.imshow(img_array)
                ax.set_yticks([])
                ax.set_xticks([])
        else:
            axes.imshow(images[0], cmap="gray")
            axes.set_yticks([])
            axes.set_xticks([])
        plt.tight_layout()
        return fig

    selected_images = (
        show_images(list(chart.value["index"]))
    )

    mo.md(
        f"""
        **Here's a preview of the images you've selected**:

        {mo.as_html(selected_images)}
        """
    )
    return selected_images, show_images


@app.cell
def _(chart):
    img_paths = [f"data/pokemon_jpg/{_}.jpg" for _ in chart.value['index']]
    return (img_paths,)


@app.cell
def _():
    import numpy as np
    return (np,)


@app.cell
def _(alt):
    def scatter(df):
    
        return (alt.Chart(df)
        .mark_circle()
        .encode(
            x=alt.X("x:Q").scale(domain=(df['x'].min(), df['x'].max())),
            y=alt.Y("y:Q").scale(domain=(df['y'].min(), df['y'].max())),
            color=alt.Color("type:N"),
        ).properties(width=500, height=500))
    return (scatter,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
