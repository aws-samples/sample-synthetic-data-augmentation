"""Prompt templates for Qwen Image Edit synthetic person insertion.

The prompt is assembled from modular blocks so the placement strategy and
optional scene augmentation can be toggled independently.
"""

PLACEMENT_HAZARDOUS = (
    "Position the person in a hazardous place or position such as on the "
    "tracks, standing on top of the train, hanging on to the edge, etc."
)

PLACEMENT_BACKGROUND = (
    "Position the person in the background of the image, far away from the train. "
    "The person should be very small compared to the rest of the image."
)

SCALE_BLOCK = (
    "Ensure the person is:\n"
    "- Accurately scaled and proportioned relative to other objects. "
    "Train cars are between 15-20 feet tall. A person should never be "
    "larger than a train car.\n"
    "- Naturally integrated into the scene's depth, perspective, and lighting.\n"
    "- Clearly visible and **not blurred**"
)

SCENE_BLOCK = (
    "Lighting and atmosphere instructions:\n"
    "- Simulate lighting appropriate for {time_of_day} time, with consistent "
    "shadows and highlights.\n"
    "- Reflect the ambient condition of {ambient_condition} in the scene's "
    "atmosphere, visibility, and color tones.\n"
    "- Avoid artificial shine or polished textures. Machinery and surfaces "
    "should appear natural, with realistic textures (e.g., dust, dirt, faded paint).\n"
    "- Colors should be realistic and muted, not overly saturated.\n"
    "- Avoid glossy or polished finishes."
)

QUALITY_BLOCK = (
    "Do not modify or distort any existing elements in the input image. "
    "The person must not appear larger or smaller than expected based on the scene's scale.\n"
    "Do not change any numbers, text, logos, or symbols on machinery or objects "
    "from the input image.\n\n"
    "Visual quality requirements:\n"
    "- Maintain **sharp focus across the entire depth of the image**, avoiding "
    "shallow depth-of-field effects.\n"
    "- Avoid: garbled face, floating or incomplete body parts, over-saturation, "
    "low resolution, grainy textures, pixelation, under/overexposure, poor color "
    "balance, washed-out tones, artifacts, color banding, outdated effects, "
    "unrealistic elements, poor compositing, visual noise, flickering, or "
    "**background blur**.\n"
    "- Ensure: high realism, consistent geometry, natural integration of all "
    "elements, **uniform sharpness across foreground and background**, "
    "**high-definition** rendering with crisp details."
)


def build_prompt(
    gender: str,
    placement: str,
    scene_augmentation: bool,
    time_of_day: str = "",
    ambient_condition: str = "",
) -> str:
    """Assemble the full edit prompt.

    :param gender: "male" or "female"
    :param placement: "hazardous" or "background"
    :param scene_augmentation: include time-of-day/ambient lighting instructions
    :param time_of_day: used only when scene_augmentation is True
    :param ambient_condition: used only when scene_augmentation is True
    """
    placement_text = PLACEMENT_HAZARDOUS if placement == "hazardous" else PLACEMENT_BACKGROUND

    parts = [
        f"Add one realistic person (gender: {gender}) to this image.",
        placement_text,
        SCALE_BLOCK,
    ]

    if scene_augmentation:
        parts.append(
            SCENE_BLOCK.format(
                time_of_day=time_of_day, ambient_condition=ambient_condition
            )
        )

    parts.append(QUALITY_BLOCK)
    return "\n\n".join(parts)
