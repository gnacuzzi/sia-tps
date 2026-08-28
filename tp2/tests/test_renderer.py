from sia_tp2.domain.model import Individual, TriangleGene
from sia_tp2.domain.renderer import render_individual, working_size


def _triangle(color):
    return TriangleGene(
        vertices=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), color=color
    )


def test_rendering_is_deterministic() -> None:
    individual = Individual(
        chromosome=(_triangle((255, 0, 0, 128)), _triangle((0, 0, 255, 128)))
    )

    first = render_individual(individual, size=(16, 16), canvas_rgb=(255, 255, 255))
    second = render_individual(individual, size=(16, 16), canvas_rgb=(255, 255, 255))

    assert first.mode == "RGB"
    assert first.tobytes() == second.tobytes()


def test_alpha_composition_changes_when_gene_order_changes() -> None:
    red = _triangle((255, 0, 0, 128))
    blue = _triangle((0, 0, 255, 128))

    red_then_blue = render_individual(
        Individual(chromosome=(red, blue)),
        size=(16, 16),
        canvas_rgb=(255, 255, 255),
    )
    blue_then_red = render_individual(
        Individual(chromosome=(blue, red)),
        size=(16, 16),
        canvas_rgb=(255, 255, 255),
    )

    assert red_then_blue.getpixel((2, 2)) != blue_then_red.getpixel((2, 2))
    assert red_then_blue.tobytes() != blue_then_red.tobytes()


def test_working_size_preserves_aspect_ratio() -> None:
    assert working_size((560, 368), 64) == (64, 42)
    assert working_size((612, 612), 64) == (64, 64)

