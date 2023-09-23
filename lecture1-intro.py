from manim import *
import numpy as np
import random

#config.background_color = "#800000"

class OpeningScene(VectorScene):
    def construct(self):

        text = Text("Review of the Syllabus", font="Consolas", font_size=70, color='YELLOW')
        difference = Text(
            """A math-heavy course with the foundations of Linear Algebra,\n specific focus on applications for Machine Learning""",
              font="Consolas", t2c={'Machine Learning':YELLOW}, font_size=24)
        VGroup(text, difference).arrange(DOWN, buff=1)
        self.play(Write(text))
        self.play(Write(difference))
        self.wait(1)

class Notation(VectorScene):
    def construct(self):
        #title = Title("Notation and Conventions")              
        #self.play(Write(title))
        #self.wait(1)
        image = ImageMobject("image.jpg")
        self.play(FadeIn(image))
        self.wait(1)
        self.play(image.animate.scale(50))
        plane = NumberPlane(
            x_axis_config={"stroke_opacity": 0},
            y_axis_config={"stroke_opacity": 0},
        )
        self.play(Create(plane))
        start_point = UP * 0.5 + RIGHT * 3.5
        pixel_label = Text("pixel intensities").move_to(start_point + DOWN).set_color(YELLOW)
        self.play(Write(pixel_label))
        numbers = VGroup() # Group to contain the numbers
        for i, num in enumerate(["0", "2", "1"]):
            number = MathTex(num).move_to(start_point - i * DOWN)
            numbers.add(number)
            self.play(Write(number))

        self.wait(1)
        # Remove the image and the grid, leaving only the numbers
        self.play(FadeOut(image), FadeOut(plane), FadeOut(pixel_label))
        self.wait(2)

        label = Text("scalar").next_to(numbers, RIGHT, buff=1).set_color(YELLOW) # Increase buff to move label further to the right
        self.play(Write(label))

        # Draw arrows from the label to each of the numbers
        arrows = VGroup()
        for number in numbers:
            arrow = Arrow(label.get_left(), number.get_right(), buff=0.1)
            arrows.add(arrow)
        self.play(Create(arrows))
        self.play(FadeOut(arrows), label.animate.to_corner(UL).shift(DOWN))
        characters = VGroup(
            MathTex("a"),
            MathTex("b"),
            MathTex(r"\alpha"),
            MathTex(r"\beta"),
        )
        characters.arrange(RIGHT, buff=0.5).next_to(label, RIGHT)
        self.play(Write(characters))
        vector = Tex(r"\[\begin{bmatrix} 0 \\ 2 \\ 1\end{bmatrix}\]").move_to(numbers)
        self.wait(1) 
        self.play(FadeOut(numbers), FadeIn(vector))
        self.wait(1) 
        label_vec = Text("vector").next_to(vector, RIGHT, buff=1).set_color(YELLOW) # Increase buff to move label further to the right
        arrow = Arrow(label_vec.get_left(), vector.get_right(), buff=0.1)
        self.play(Write(label_vec), Create(arrow))
        self.wait(1)
        self.play(label_vec.animate.next_to(label, DOWN), FadeOut(arrow))
        characters = VGroup(
            MathTex("\mathbf{x}"),
            MathTex("\mathbf{y}"),
        )
        characters.arrange(RIGHT, buff=0.5).next_to(label_vec, RIGHT)
        self.wait(1)
        self.play(Write(characters))
        self.wait(1)
        explanation_text = MathTex(
            r"x_i \text{ (}i^{th}\text{ element of vector } \mathbf{x}) \text{ e.g. } x_2 = 2"
        ).next_to(label_vec, DOWN).shift(3.5*RIGHT)

        self.play(Write(explanation_text), vector[0][3].animate.set_color(YELLOW))
        self.wait(1)
        matrix = MathTex(r"\begin{bmatrix} 0 & 0 \\ 2 & 1 \\ 1 & 2 \end{bmatrix}")

        # Position the new matrix where the old vector was
        matrix.move_to(vector)

        # Animate the transition from the vector to the matrix
        self.play(Transform(vector, matrix))

        self.wait(1)
        label_mat = Text("matrix").next_to(matrix, 0.5*RIGHT, buff=1).set_color(YELLOW) # Increase buff to move label further to the right
        arrow = Arrow(label_mat.get_left(), matrix.get_right(), buff=0.1)
        self.play(Write(label_mat), Create(arrow))
        self.wait(1)        
        self.play(label_mat.animate.next_to(explanation_text, DOWN).shift(3.5*LEFT), FadeOut(arrow))
        self.wait(1)
        characters = VGroup(
            MathTex("\mathbf{A}"),
            MathTex("\mathbf{B}"),
        )
        characters.arrange(RIGHT, buff=0.5).next_to(label_mat, RIGHT)
        self.wait(1)
        self.play(Write(characters))
        self.wait(1)
        explanation_text = MathTex(
            r"A_{ij} \text{ (}i,j^{th}\text{ element of matrix } \mathbf{A}) \text{ e.g. } A_{21} = 1"
        ).next_to(label_mat, DOWN).shift(4*RIGHT)

        self.play(Write(explanation_text), vector[0][6].animate.set_color(YELLOW))
        self.wait(1)

class VectorDefinition(VectorScene):
    def construct(self):
        #title = Title("Vectors")  
        plane = NumberPlane(
            x_range=[-3, 3], 
            y_range=[-3, 3], 
            axis_config={"color": WHITE},
            background_line_style={
                "stroke_color": WHITE,
                "stroke_width": 2,
                "stroke_opacity": 0.5
            }
        )          
        #self.play(Write(title))
        #self.wait(1)
        x = Vector([1, 2]).set_color(WHITE)
        x_label = MathTex("\mathbf{x}").next_to(x, RIGHT)
        x_value = Tex(r"\[\begin{bmatrix} 1 \\ 2\end{bmatrix}\]").next_to(x_label, RIGHT)

        y = Vector([-2, 1]).set_color(YELLOW)
        y_label = MathTex("\mathbf{y}").next_to(y, LEFT)
        y_value = Tex(r"\[\begin{bmatrix} -2 \\ 1\end{bmatrix}\]").next_to(y_label, LEFT)
        
        # Add the vectors and labels to the scene
        self.play(Create(plane))
        self.play(Create(x), Write(x_label), Write(x_value))
        self.play(Create(y), Write(y_label), Write(y_value))
        self.wait(2)

        ball_x = Dot(point=[1,2,0], color=WHITE, radius=0.2)
        ball_y = Dot(point=[-2,1,0], color=YELLOW, radius=0.2)
        text = Text("Vector as Points on a Plane", font="Comic Sans MS", 
                    font_size=20, 
                    color='LIGHTGREEN').next_to(x_value, RIGHT)
        # Immediately try fading out the balls
        self.play(FadeIn(ball_x), FadeIn(ball_y), FadeOut(x), FadeOut(y), Write(text))
        self.wait(2)

        self.play(FadeOut(ball_x), FadeOut(ball_y), FadeIn(x), FadeIn(y), FadeOut(text))
        self.wait(2)
        angle_between_x = angle_between_vectors([1, 0], [1, 2])
        angle_x = Sector(start_angle=0, angle=angle_between_x, radius=0.35, color="LIGHTBLUE")
        theta_x_label = MathTex(r"\theta_x").next_to(angle_x, RIGHT, buff=0.2)
        r_x_label = MathTex(r"||x||").next_to(x.get_center(), UP).shift(0.2*LEFT)

        # Angle between x-axis and vector y
        angle_between_y = angle_between_vectors([-1, 0], [-2, 1])

        # Check for second quadrant
        #if -2 < 0 and 1 > 0:
        #    angle_between_y = 2 * np.pi - angle_between_y

        angle_y = Sector(start_angle=np.pi, angle=-angle_between_y, radius=0.15, color="LIGHTGREEN")
        theta_y_label = MathTex(r"\theta_y").next_to(angle_y, LEFT, buff=0.2)
        r_y_label = MathTex(r"||y||").next_to(y.get_center(), RIGHT).shift(0.2*UP)

        self.play(Create(angle_x), Create(angle_y), Write(theta_x_label),
                   Write(theta_y_label), Write(r_x_label), Write(r_y_label),
                   FadeOut(x_label), FadeOut(x_value),
                   FadeOut(y_label), FadeOut(y_value))
        self.wait(4)
        self.play(FadeOut(angle_x), FadeOut(angle_y), FadeOut(theta_x_label),
                   FadeOut(theta_y_label), FadeOut(r_x_label), FadeOut(r_y_label),
                   Write(x_label), Write(x_value),
                   Write(y_label), Write(y_value))
        self.wait(2)



class ScalarMultiplication(VectorScene):
    def construct(self):
        #title = Title("Vector Operations")  
        plane = NumberPlane(
            x_range=[-3, 3], 
            y_range=[-3, 3], 
            axis_config={"color": WHITE},
            background_line_style={
                "stroke_color": WHITE,
                "stroke_width": 2,
                "stroke_opacity": 0.5
            }
        )          
        #self.play(Write(title))
        #self.wait(1)
        x = Vector([1, 2]).set_color(WHITE)
        x_label = MathTex("\mathbf{x}").next_to(x, RIGHT)
        x_value = Tex(r"\[\begin{bmatrix} 1 \\ 2\end{bmatrix}\]").next_to(x_label, RIGHT)
        shift_distance = LEFT * 3
        #y = Vector([-2, 1]).set_color(YELLOW).shift(shift_distance)
        #y_label = MathTex("\mathbf{y}").next_to(y, LEFT)
        #y_value = Tex(r"\[\begin{bmatrix} -2 \\ 1\end{bmatrix}\]").next_to(y_label, LEFT)
        #self.play(Create(y), Write(y_label), Write(y_value))
        self.play(Create(plane), Create(x), Write(x_label), Write(x_value))
        self.wait(2)
        
        self.play(
            ApplyMethod(plane.shift, shift_distance),
            ApplyMethod(x.shift, shift_distance),
            ApplyMethod(x_label.shift, shift_distance),
            ApplyMethod(x_value.shift, shift_distance)
        )
        self.wait(1)
                # Add title "Scalar Multiplication" on the right side
        scalar_title = Text("Scalar Multiplication", 
                            color="LIGHTGREEN",
                            font="Comic Sans MS", 
                            font_size=20).next_to(x_label, RIGHT).shift(RIGHT*3).shift(UP*2)
        self.play(Write(scalar_title))
        self.wait(2)
        x_multiplied_text = MathTex(r"a\mathbf{x} = \begin{bmatrix} a x_1 \\ a x_2 \end{bmatrix}").next_to(scalar_title, DOWN)
        self.play(Write(x_multiplied_text))

        # Show multiplication of vector x with scalar a=2
        self.wait(1)
        scalar_display_position = x_multiplied_text.get_bottom() + 0.5 * DOWN
        x_multiplied_old = Vector([1 * (-1.5), 2 * (-1.5)]).set_color(RED).shift(shift_distance)
        #x_original = Vector([1, 2]).set_color(WHITE).shift(shift_distance)
        scalar_value_old = MathTex(f"a = -1.5").move_to(scalar_display_position)
        for a in np.linspace(-1.5, 1.5, 40):  # smoothly transition from -1.5 to 1.5
            scalar_value = MathTex(f"a = {a:.1f}").move_to(scalar_display_position)
            x_multiplied = Vector([1 * a, 2 * a]).set_color(RED).shift(shift_distance)
            self.play(ReplacementTransform(x_multiplied_old, x_multiplied), TransformMatchingTex(scalar_value_old, scalar_value), run_time=0.1)
            scalar_value_old = scalar_value
            x_multiplied_old = x_multiplied
        #self.play(FadeOut(x_original))
        self.play(FadeOut(scalar_value), FadeOut(x_multiplied))
        self.wait(2)
        scalar_title2 = Text("Now let's see Scalar Multiplication\nin data points", 
                    color="LIGHTGREEN",
                    font="Comic Sans MS", 
                    font_size=20).next_to(x_label, RIGHT).shift(RIGHT*3).shift(UP*2)
        ball_x = Dot(point=[1,2,0], color=WHITE, radius=0.1).shift(shift_distance)
        self.play(Transform(scalar_title, scalar_title2))
        self.wait(1)
        self.play(FadeIn(ball_x), FadeOut(x))
        self.wait(2)
        scalar_value_old = MathTex(f"a = -1.5").move_to(scalar_display_position)
        a = -1.5
        ball_x_old = Dot(point=[1*a,2*a,0], color=RED, radius=0.1).shift(shift_distance)
        for a in np.linspace(-1.5, 1.5, 40):  # smoothly transition from -1.5 to 1.5
            scalar_value = MathTex(f"a = {a:.1f}").move_to(scalar_display_position)
            ball_x_new = Dot(point=[1*a,2*a,0], color=RED, radius=0.1).shift(shift_distance)
            self.play(ReplacementTransform(ball_x_old, ball_x_new), TransformMatchingTex(scalar_value_old, scalar_value), run_time=0.1)
            scalar_value_old = scalar_value
            ball_x_old = ball_x_new
        self.wait(2)
                
        line = Line([-2*1, -2*2, 0], [2*1, 2*2, 0]).set_color(ORANGE).shift(shift_distance)
        scalar_text = MathTex(r"\text{Points of }", 
                      "a", 
                      r"\mathbf{x}", 
                      r"\\ \text{ for all values of }", 
                      "a", 
                      r"\\ \text{ form a line}")

        scalar_text.next_to(scalar_value, DOWN)
        self.play(Create(line), FadeOut(ball_x_new), Write(scalar_text))
        self.wait(2)

class VectorAddition(VectorScene):
    def construct(self):
        shift_distance = LEFT * 3
        plane = NumberPlane(
            x_range=[-3, 3], 
            y_range=[-3, 3], 
            axis_config={"color": WHITE},
            background_line_style={
                "stroke_color": WHITE,
                "stroke_width": 2,
                "stroke_opacity": 0.5
            }
        ).shift(shift_distance)        

        x = Vector([1, 2]).set_color(WHITE).shift(shift_distance) 
        x_label = MathTex("\mathbf{x}").next_to(x, RIGHT)
        x_value = Tex(r"\[\begin{bmatrix} 1 \\ 2\end{bmatrix}\]").next_to(x_label, RIGHT)
        
        y = Vector([-2, -1]).set_color(YELLOW).shift(shift_distance)
        y_label = MathTex("\mathbf{y}").next_to(y, LEFT)
        y_value = Tex(r"\[\begin{bmatrix} -2 \\ -1\end{bmatrix}\]").next_to(y_label, LEFT)
        self.play(Create(plane), Create(x), Write(x_label), Write(x_value))      
        self.play(Create(y), Write(y_label), Write(y_value))
        self.wait(2)
        
                # Add title "Scalar Multiplication" on the right side
        vector_title = Text("Vector Addition", 
                            color="LIGHTGREEN",
                            font="Comic Sans MS", 
                            font_size=20).next_to(x_label, RIGHT).shift(RIGHT*3).shift(UP*2)
        self.play(Write(vector_title))
        self.wait(2)
        # Sum of the vectors
        z = Vector([-1, 1]).set_color(BLUE).shift(shift_distance)      
        z_value = Tex(r"\[\begin{bmatrix} -1 \\ 1\end{bmatrix}\]").next_to(z, LEFT)
        z_label = MathTex("\mathbf{z}").next_to(z_value, LEFT)

        # Displaying z = x + y
        sum_label = MathTex("\mathbf{z} = \mathbf{x} + \mathbf{y}").next_to(vector_title, DOWN)
        new_start_point = x.get_end()
        new_end_point = z.get_end()
        # Move y to the new position
        self.play(Write(sum_label))
        self.wait(1)
        self.play(y.animate.put_start_and_end_on(new_start_point, new_end_point),
                   y_label.animate.shift(3*UP + 3*RIGHT), y_value.animate.shift(3*UP + 3*RIGHT))
        self.wait(2)
        
        # Breaking down the addition component-wise
        componentwise_addition = MathTex(r"""

        \begin{bmatrix}
        1 \\
        2
        \end{bmatrix}
        +
        \begin{bmatrix}
        -2 \\
        -1
        \end{bmatrix}
        =
        \begin{bmatrix}
        -1 \\
        1
        \end{bmatrix}
        """).next_to(sum_label, DOWN)

        self.play(Write(componentwise_addition))
        self.play(Create(z), Write(z_label), Write(z_value))

        self.wait(2)
        
class VectorSubtraction(VectorScene):
    def construct(self):
        shift_distance = LEFT * 3
        plane = NumberPlane(
            x_range=[-3, 3], 
            y_range=[-3, 3], 
            axis_config={
                "color": WHITE,
                "include_numbers": True,  # This will include numbers on the axis
            },
            background_line_style={
                "stroke_color": WHITE,
                "stroke_width": 2,
                "stroke_opacity": 0.5
            }
        ).shift(shift_distance)

        x = Vector([1, 2]).set_color(WHITE).shift(shift_distance) 
        x_label = MathTex("\mathbf{x}").next_to(x, RIGHT)
        y = Vector([-2, -1]).set_color(YELLOW).shift(shift_distance)
        y_label = MathTex("\mathbf{y}").next_to(y, LEFT)
        self.play(Create(x), Write(x_label), Create(y), Create(plane), Write(y_label))
        self.wait(1)

        vector_title = Text("Vector Subtraction", 
                        color="LIGHTGREEN",
                        font="Comic Sans MS", 
                        font_size=20).next_to(x, RIGHT).shift(RIGHT*3).shift(UP*2)
        self.play(Write(vector_title))

        self.wait(1)

        sub_label = MathTex("\mathbf{z} = \mathbf{x} - \mathbf{y}").next_to(vector_title, DOWN)
        self.play(Write(sub_label))

        z_start = y.get_end()
        z_end = x.get_end()
        z = Vector(z_end - z_start).shift(z_start).set_color(RED)
        componentwise_subtraction = MathTex(r"""

        \begin{bmatrix}
        1 \\
        2
        \end{bmatrix}
        -
        \begin{bmatrix}
        -2 \\
        -1
        \end{bmatrix}
        =
        \begin{bmatrix}
        3 \\
        3
        \end{bmatrix}
        """).next_to(sub_label, DOWN)
        self.play(Create(z), Write(componentwise_subtraction))
        self.wait(1)
        z_shifted = Vector(z_end - z_start).shift(shift_distance)
        self.play(z.animate.put_start_and_end_on(x.get_start(), z_shifted.get_end()))
        self.wait(2)
        x_end = x.get_end()
        # Rotating y to match x and animating z vector
        target_angle = np.arctan2(x_end[1], x_end[0])
        start_angle = np.arctan2(z_start[1], z_start[0])
        rotation_steps = np.linspace(start_angle, target_angle, 40)  # Adjust 40 for more or fewer steps

        for angle in rotation_steps:
            rotated_y = np.array([np.cos(angle) * np.sqrt(5), np.sin(angle) * np.sqrt(5), 0])
            new_y = Vector(rotated_y).shift(shift_distance).set_color(YELLOW)

            new_z_start = new_y.get_end()
            new_z_end = x.get_end()
            new_z = Vector(new_z_end - new_z_start).shift(new_z_start).set_color(RED)

            updated_subtraction = MathTex(rf"""
            \begin{{bmatrix}}
            1 \\
            2
            \end{{bmatrix}}
            -
            \begin{{bmatrix}}
            {rotated_y[0]:.1f} \\
            {rotated_y[1]:.1f}
            \end{{bmatrix}}
            =
            \begin{{bmatrix}}
            {1 - rotated_y[0]:.1f} \\
            {2 - rotated_y[1]:.1f}
            \end{{bmatrix}}
            """).next_to(sub_label, DOWN)
            
            self.play(
                ReplacementTransform(y, new_y), 
                ReplacementTransform(z, new_z), 
                ReplacementTransform(componentwise_subtraction, updated_subtraction), 
                run_time=0.1
            )

            y = new_y
            z = new_z
            componentwise_subtraction = updated_subtraction

        self.wait(2)

        new_y = Vector([1, 2]).shift(shift_distance).set_color(YELLOW)
        new_z_start = new_y.get_end()
        new_z_end = x.get_end()
        new_z = Vector(new_z_end - new_z_start).shift(new_z_start).set_color(RED)

        updated_subtraction = MathTex(rf"""
        \begin{{bmatrix}}
        1 \\
        2
        \end{{bmatrix}}
        -
        \begin{{bmatrix}}
        1 \\
        2
        \end{{bmatrix}}
        =
        \begin{{bmatrix}}
        0 \\
        0
        \end{{bmatrix}}
        """).next_to(sub_label, DOWN)
        
        self.play(
            ReplacementTransform(y, new_y), 
            ReplacementTransform(z, new_z), 
            ReplacementTransform(componentwise_subtraction, updated_subtraction), 
            run_time=1
        )
        self.wait(2)
        zero_vector_title = Text("Zero Vector", 
                color="ORANGE",
                font="Comic Sans MS", 
                font_size=25).next_to(updated_subtraction, DOWN)#.shift(RIGHT*3).shift(UP*2)
        self.play(Write(zero_vector_title))
        self.wait(2)


class LinearCombination(ThreeDScene):
    def construct(self):
        shift_distance = LEFT * 3
        plane = NumberPlane(
            x_range=[-3, 3], 
            y_range=[-3, 3], 
            axis_config={
                "color": WHITE,
                "include_numbers": True,  # This will include numbers on the axis
            },
            background_line_style={
                "stroke_color": WHITE,
                "stroke_width": 2,
                "stroke_opacity": 0.5
            }
        ).shift(shift_distance)

        # Create the basis vectors
        x = Vector([2, 1], color=BLUE).shift(shift_distance)
        y = Vector([1, -2], color=RED).shift(shift_distance)

        x_label = MathTex("\mathbf{x}").next_to(x, RIGHT).set_color(BLUE)
        y_label = MathTex("\mathbf{y}").next_to(y, LEFT).set_color(RED)

        # Define scalars and resulting linear combination

        # Display everything
        self.play(Create(plane), Create(x), Write(x_label), Create(y), Write(y_label))
        self.wait(1)


        # LaTeX Descriptions
        description_1 = MathTex("Given~two~vectors~", "\mathbf{x}",
                                 "~and~", "\mathbf{y},").next_to(plane, RIGHT, aligned_edge=LEFT).shift(RIGHT*3).shift(UP*3)

        description_2 = MathTex("a~linear~combination~is:").next_to(description_1, DOWN, aligned_edge=LEFT).set_color(YELLOW)

        equation_1 = MathTex("a\mathbf{x} + b\mathbf{y}").next_to(description_2, DOWN, aligned_edge=LEFT)

        description_2_5 = MathTex("where,~", "a~and~b~are~scalars", ".").next_to(equation_1, DOWN, aligned_edge=LEFT)

        description_3 = MathTex("For~more~vectors,~").next_to(description_2_5, DOWN, aligned_edge=LEFT)
        description_3_5 = MathTex("\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_n", ",").next_to(description_3, DOWN, aligned_edge=LEFT)
        equation_2 = MathTex("a_1\mathbf{x}_1 + a_2\mathbf{x}_2 + \dots + a_n\mathbf{x}_n").next_to(description_3_5, DOWN, aligned_edge=LEFT)

        self.play(
            Write(description_1),
            Write(description_2),
            Write(equation_1),
            Write(description_2_5))
        self.wait(1)
        self.play(Write(description_3),
            Write(description_3_5),
            Write(equation_2))
        self.wait(1)
        a_values = [-1, -.5, 0, .5, 1]
        b_values = [-1, -.5, 0, .5, 1]
        old_vector = Vector(x.get_end() + y.get_end(), color=YELLOW).shift(shift_distance)
        ab_text = MathTex(rf"a = -1, b = -1 ").next_to(equation_2, 2*DOWN).set_color(YELLOW)
        self.play(Write(ab_text))
        for a in a_values:
            for b in b_values:
                resultant_vector = a * np.asarray([2,1]) + b * np.asarray([1,-2])
                lin_comb = Vector(resultant_vector, color=YELLOW).shift(shift_distance)


                updated_ab_text = MathTex(rf"a = {a}, b = {b}").next_to(equation_2, 2*DOWN).set_color(YELLOW)
                
                self.play(
                    ReplacementTransform(old_vector, lin_comb), 
                    ReplacementTransform(ab_text, updated_ab_text), 
                    run_time=0.5
                )

                old_vector = lin_comb
                ab_text = updated_ab_text

        self.wait(2)   


class LinearIndependence(ThreeDScene):
    def construct(self):

        definition1 = MarkupText(
            r"A set of vectors is <span fgcolor='#FFFF00'>linearly independent</span> if no vector in the set", font_size=24
        ).to_edge(UP)#.shift(DOWN)
        definition2 = Text(
            "can be written as a linear combination of the other vectors.", font_size=24
        )

        equation = MathTex(
            "c_1 ", "\\mathbf{x}_1 ", "+ c_2 ", "\\mathbf{x}_2 ",
            "+ \\ldots + c_k ", "\\mathbf{x}_k ", "= ", "\\mathbf{0}"
        )

        condition = MathTex(
            "c_1 = c_2 = \\ldots = c_k = 0"
        )

        note = Text(
            "If any non-zero coefficients exist satisfying the equation, the vectors are linearly dependent.", font_size=24
        )


        definition2.next_to(definition1, DOWN)
        equation.next_to(definition2, DOWN).shift(0.5*DOWN)
        condition.next_to(equation, DOWN)
        note.next_to(condition, DOWN)#.shift(0.5*DOWN)

        # Adding all components to the scene
        self.wait(1)
        self.play(Write(definition1), Write(definition2))
        self.wait(1)
        self.play(Write(equation))
        self.wait(1)
        self.play(Write(condition))
        self.wait(1)
        self.play(Write(note))
        self.wait(4)
        example_text = MathTex(
            r"e.g.\ x_1 = \begin{bmatrix} 1 \\ 2 \\ 1 \end{bmatrix},\ x_2 = \begin{bmatrix} 3 \\ 0 \\ 1 \end{bmatrix}\ \text{and}\ x_3 = \begin{bmatrix} 1 \\ -4 \\ -1 \end{bmatrix}",
            r"\text{ are not linearly independent; because }",
            r"-2x_1 + x_2 = x_3."
        )
        
        # Aligning the equation properly
        example_text.arrange_submobjects(DOWN, aligned_edge=LEFT, buff=0.25).next_to(note, DOWN)
        
        # Adding the equation to the scene
        self.play(Write(example_text), run_time=5)
        self.wait(4)
        self.play(FadeOut(example_text), FadeOut(note), FadeOut(condition), FadeOut(equation), FadeOut(definition1),FadeOut(definition2) )
        question = Text("OK - but, why is linear independence important?", 
                        color="LIGHTGREEN",
                        font="Comic Sans MS", 
                        font_size=25)
        self.play(Write(question))
        self.wait(4)

class ThreeDScene1(ThreeDScene):
    def construct(self):       
        axes = ThreeDAxes()
        self.add(axes)

        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.begin_ambient_camera_rotation(rate=75 * DEGREES / 4)
        self.wait(2)
        x = np.array([1, 0, 0])
        line = Line(-5 * x, 5 * x, color=YELLOW)
        self.add(line)
        self.wait(5)
        

class ThreeDScene2(ThreeDScene):
    def construct(self):       
        axes = ThreeDAxes()
        self.add(axes)
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.begin_ambient_camera_rotation(rate=75 * DEGREES / 4)
        self.wait(2)

        x = np.array([1, 0, 0])
        y = np.array([0, 1, 0])

        # Compute the vertices of the parallelogram
        v1 = -4 * x + -4 * y
        v2 = -4 * x + 4 * y
        v3 = 4 * x + 4 * y
        v4 = 4 * x + -4 * y
        #Create the parallelogram using Polygon mobject
        hyperplane = Polygon(v1, v2, v3, v4, color=BLUE, fill_opacity=0.5)
        self.add(hyperplane)
        self.wait(8)
class ThreeDScene3(ThreeDScene):
    def construct(self):       
        axes = ThreeDAxes()
        self.add(axes)
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.begin_ambient_camera_rotation(rate=75 * DEGREES / 4)
        self.wait(2)

        x = np.array([1, 0, 0])
        y = np.array([0, 0, 1])

        # Compute the vertices of the parallelogram
        v1 = -4 * x + -4 * y
        v2 = -4 * x + 4 * y
        v3 = 4 * x + 4 * y
        v4 = 4 * x + -4 * y
        #Create the parallelogram using Polygon mobject
        hyperplane = Polygon(v1, v2, v3, v4, color=BLUE, fill_opacity=0.5)
        self.add(hyperplane)
        self.wait(8)

class ThreeDScene4(ThreeDScene):
    def construct(self):  
        axes = ThreeDAxes()
        self.add(axes)
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.begin_ambient_camera_rotation(rate=75 * DEGREES / 4)
        self.wait(2)
        cube = Cube(color=BLUE, fill_opacity=0.5).scale(4)
        self.add(cube)
        self.wait(5)

class ThreeDScene5(ThreeDScene):
    def construct(self):       
        axes = ThreeDAxes()
        self.add(axes)
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.begin_ambient_camera_rotation(rate=75 * DEGREES / 4)
        self.wait(2)
        x = np.array([3, 0, 1])
        line = Line(-5 * x, 5 * x, color=YELLOW)
        self.add(line)
        self.wait(5)

class ThreeDScene6(ThreeDScene):
    def construct(self):       
        axes = ThreeDAxes()
        self.add(axes)
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.begin_ambient_camera_rotation(rate=75 * DEGREES / 4)
        self.wait(2)

        x = np.array([3, 0, 1])
        y = np.array([1, 2, 1])

        # Compute the vertices of the parallelogram
        v1 = -2 * x + -2 * y
        v2 = -2 * x + 2 * y
        v3 = 2 * x + 2 * y
        v4 = 2 * x + -2 * y
        #Create the parallelogram using Polygon mobject
        hyperplane = Polygon(v1, v2, v3, v4, color=GREEN, fill_opacity=0.4)
        self.add(hyperplane)
        self.wait(12)
        
class InnerProduct(VectorScene):
    def construct(self):
        
        vect1 = Matrix([[1, 3, 5]]).shift(LEFT*2)
        dot = Dot().next_to(vect1, RIGHT)
        vect2 = Matrix([[2,], [4,], [6,]]).next_to(dot, RIGHT)
        
        vectGroup = VGroup(vect1, dot, vect2)
        
        self.play(FadeIn(vectGroup), run_time=4)
        equal = Text(" = ").next_to(vectGroup, RIGHT)          
        self.play(Write(equal))
        
        # Highlighting the first entries with rectangles
        highlight_color = BLUE  # Choose any color you like
        rect1 = SurroundingRectangle(vect1[0][0], color=highlight_color)
        rect2 = SurroundingRectangle(vect2[0][0], color=highlight_color)
        self.play(Create(rect1), Create(rect2))
        colored_prod = MathTex(r"1 \cdot 2", color=highlight_color).next_to(equal, RIGHT)
        self.play(Write(colored_prod))
        plus = Text(" + ").next_to(colored_prod, RIGHT)  
        self.play(Write(plus))
        self.wait()

        highlight_color = YELLOW  # Choose any color you like
        rect1 = SurroundingRectangle(vect1[0][1], color=highlight_color)
        rect2 = SurroundingRectangle(vect2[0][1], color=highlight_color)
        self.play(Create(rect1), Create(rect2))
        colored_prod = MathTex(r"3 \cdot 4", color=highlight_color).next_to(plus, RIGHT)
        self.play(Write(colored_prod))
        plus = Text(" + ").next_to(colored_prod, RIGHT)  
        self.play(Write(plus))
        self.wait()
    
        highlight_color = RED  # Choose any color you like
        rect1 = SurroundingRectangle(vect1[0][2], color=highlight_color)
        rect2 = SurroundingRectangle(vect2[0][2], color=highlight_color)
        self.play(Create(rect1), Create(rect2))
        colored_prod = MathTex(r"5 \cdot 6", color=highlight_color).next_to(plus, RIGHT)
        self.play(Write(colored_prod))
        plus = Text(" + ").next_to(colored_prod, RIGHT)  
        text = Text("(scalar)", font_size=26, color=GREEN).next_to(colored_prod, DOWN).shift(LEFT).shift(DOWN)
        soln = Text("44", font_size=26, color=YELLOW).next_to(text, DOWN)
        self.play(Write(text), Create(soln))
        self.wait(2)


class matrix_vector_mult(Scene):
    def construct(self):
        A = np.array([[-1, 2], [2, 0]])
        B = np.array([[2], [1]])
        C = np.dot(A, B)
        stuff = VGroup(Matrix(A), Matrix(B), Matrix(C))
        matrixA = stuff[0]
        matrixB = stuff[1]
        matrixC = stuff[2]
        matrixA.height = 2.5
        matrixB.height = 2.5
        matrixC.height = 2.5
        Dot = Tex(".", color=WHITE, font_size = 200)
        Equals = Tex("=", color=WHITE, font_size = 100)
        bOpen = Tex("[", color=WHITE, font_size = 100)
        bClose = Tex("]", color=WHITE, font_size=100)
        bOpen1 = Tex("[", color=WHITE, font_size=200)
        bClose1 = Tex("]", color=WHITE, font_size=200)
        
        self.play(Write(matrixA))
        self.play(matrixA.animate.scale(1).to_corner(UP+LEFT*2))
        
        Dot.next_to(matrixA, RIGHT)
        #self.play(Write(Dot))
        self.play(Write(matrixB))
        self.play(matrixB.animate.scale(1).next_to(matrixA, RIGHT))
        Equals.next_to(matrixB, RIGHT)
        self.play(Write(Equals))
        matrixC.next_to(Equals)
        #self.play(Write(matrixC))
        C_elements = VGroup(*matrixC)
        for i in C_elements[1:]:
            i.height = 2.5
            self.play(Write(i))
        
        C_elements = VGroup(*C_elements[0])

        highlight_color = BLUE
        first_row = VGroup(matrixA.get_rows()[0])
        first_column = VGroup(matrixB.get_columns()[0])
        rect1 = SurroundingRectangle(first_row, color=highlight_color)
        rect2 = SurroundingRectangle(first_column, color=highlight_color)
        self.play(Create(rect1), Create(rect2))
        _bOpen ,_bClose, _Dot, _bOpen1, _bClose1 = bOpen.copy() ,bClose.copy(), Dot.copy(), bOpen1.copy(), bClose1.copy()
        _bOpen.next_to(matrixA, 3*DOWN + LEFT)
        r = first_row.copy().next_to(_bOpen, RIGHT).set_color(BLUE)
        _bClose.next_to(r, RIGHT)
        _Dot.next_to(_bClose, RIGHT)
        _bOpen1.next_to(_Dot, RIGHT)
        c = first_column.copy().next_to(_bOpen1, RIGHT).set_color(BLUE)
        _bClose1.next_to(c, RIGHT)
        g = VGroup(_bOpen, r, _bClose, _Dot, _bOpen1, c, _bClose1).scale(0.75)
        self.play(Write(g))
        self.wait(1)
        #C_elements[0].font_size = 60
        self.play(Transform(g, C_elements[0]))
        #self.play(C_elements[0].animate.scale(1.3))
        self.play(FadeOut(rect1), FadeOut(rect2))
        highlight_color = YELLOW
        second_row = VGroup(matrixA.get_rows()[1])
        first_column = VGroup(matrixB.get_columns()[0])
        rect1 = SurroundingRectangle(second_row, color=highlight_color)
        rect2 = SurroundingRectangle(first_column, color=highlight_color)
        self.play(Create(rect1), Create(rect2))
        self.wait(2)
        _bOpen ,_bClose, _Dot, _bOpen1, _bClose1 = bOpen.copy() ,bClose.copy(), Dot.copy(), bOpen1.copy(), bClose1.copy()
        _bOpen.next_to(matrixA, 3*DOWN + LEFT)
        r = second_row.copy().next_to(_bOpen, RIGHT).set_color(highlight_color)
        _bClose.next_to(r, RIGHT)
        _Dot.next_to(_bClose, RIGHT)
        _bOpen1.next_to(_Dot, RIGHT)
        c = first_column.copy().next_to(_bOpen1, RIGHT).set_color(highlight_color)
        _bClose1.next_to(c, RIGHT)
        g = VGroup(_bOpen, r, _bClose, _Dot, _bOpen1, c, _bClose1).scale(0.75)
        self.play(Write(g))
        self.wait(1)
        #C_elements[1].font_size = 60
        self.play(Transform(g, C_elements[1]))
        #self.play(C_elements[1].animate.scale(1.3))
        self.play(FadeOut(rect1), FadeOut(rect2))
        self.wait(2)


class matrix_matrix_mult(Scene):
    def construct(self):
        A = np.array([[-1, 2], [2, 0]])
        B = np.array([[2, 0], [1, 3]])
        C = np.dot(A, B)
        stuff = VGroup(Matrix(A), Matrix(B), Matrix(C))
        matrixA = stuff[0]
        matrixB = stuff[1]
        matrixC = stuff[2]
        matrixA.height = 2.5
        matrixB.height = 2.5
        matrixC.height = 2.5
        Dot = Tex(".", color=WHITE, font_size = 200)
        Equals = Tex("=", color=WHITE, font_size = 100)
        bOpen = Tex("[", color=WHITE, font_size = 100)
        bClose = Tex("]", color=WHITE, font_size=100)
        bOpen1 = Tex("[", color=WHITE, font_size=200)
        bClose1 = Tex("]", color=WHITE, font_size=200)
        
        self.play(Write(matrixA))
        self.play(matrixA.animate.scale(1).to_corner(UP+LEFT*2))
        
        Dot.next_to(matrixA, RIGHT)
        #self.play(Write(Dot))
        self.play(Write(matrixB))
        self.play(matrixB.animate.scale(1).next_to(matrixA, RIGHT))
        Equals.next_to(matrixB, RIGHT)
        self.play(Write(Equals))
        matrixC.next_to(Equals)
        #self.play(Write(matrixC))
        C_elements = VGroup(*matrixC)
        for i in C_elements[1:]:
            i.height = 2.5
            self.play(Write(i))
        
        C_elements = VGroup(*C_elements[0])
        
        for row in range(2):
            for col in range(2):
                c_index = row*2 + col
                colors = [BLUE, YELLOW, GREEN, RED]
                highlight_color = colors[c_index]
                _row = VGroup(matrixA.get_rows()[row])
                _column = VGroup(matrixB.get_columns()[col])
                rect1 = SurroundingRectangle(_row, color=highlight_color)
                rect2 = SurroundingRectangle(_column, color=highlight_color)
                self.play(Create(rect1), Create(rect2))
                _bOpen ,_bClose, _Dot, _bOpen1, _bClose1 = bOpen.copy() ,bClose.copy(), Dot.copy(), bOpen1.copy(), bClose1.copy()
                _bOpen.next_to(matrixA, 3*DOWN + LEFT)
                r = _row.copy().next_to(_bOpen, RIGHT).set_color(highlight_color)
                _bClose.next_to(r, RIGHT)
                _Dot.next_to(_bClose, RIGHT)
                _bOpen1.next_to(_Dot, RIGHT)
                c = _column.copy().next_to(_bOpen1, RIGHT).set_color(highlight_color)
                _bClose1.next_to(c, RIGHT)
                g = VGroup(_bOpen, r, _bClose, _Dot, _bOpen1, c, _bClose1).scale(0.75)
                self.play(Write(g))
                self.wait(1)
                self.play(Transform(g, C_elements[c_index]))
                self.play(FadeOut(rect1), FadeOut(rect2))


from manim import *
import numpy as np

class LinearTransformation(Scene):
    def construct(self):
        # Define the matrix and vector
        matrix = np.array([[2, 1], [1, 2]])

        # Create and position the matrix label
        matrix_tex = MathTex(
            r"A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}"
        )
        matrix_tex.to_corner(UL)
        exp_text = Text("Let's see the effect of matrix A", font="Comic Sans MS", 
                    font_size=24, 
                    color='LIGHTGREEN').to_corner(UR)

        # Create a coordinate system
        axes = NumberPlane(
            x_range=[-5, 5], 
            y_range=[-5, 5], 
            axis_config={"color": WHITE},
            background_line_style={
                "stroke_color": WHITE,
                "stroke_width": 2,
                "stroke_opacity": 0.1
            })
        
        self.add(axes)
        self.wait(1)
        self.play(Write(matrix_tex))
        self.play(Write(exp_text))

        vectors = [np.array([1, 0]), np.array([0, 1]), np.array([-1, 2])]
        colors = [GREEN, RED, PURPLE]

        original_vectors = []
        transformed_vectors = []

        for color, v_coords in zip(colors, vectors):

            v_transformed_coords = np.dot(matrix, v_coords)
            vector_tex = MathTex(
                rf"\begin{{bmatrix}} {v_coords[0]} \\ {v_coords[1]} \end{{bmatrix}}"
            )
            vector_tex.next_to(matrix_tex)
            result_tex = MathTex(
                rf"=\begin{{bmatrix}} {v_transformed_coords[0]} \\ {v_transformed_coords[1]} \end{{bmatrix}}"
            ).next_to(vector_tex)
            vect = Arrow(start=ORIGIN, end=[*v_coords, 0], color=color, buff=0)
            vect_transformed = Arrow(start=ORIGIN, end=[*v_transformed_coords, 0], color=color, buff=0)
            v_orig = vect.copy()
            original_vectors.append(v_orig)
            transformed_vectors.append(vect_transformed)
            self.play(Create(vect), Write(vector_tex))
            self.wait(1)
            new_text = Text(f"({v_coords[0]}, {v_coords[1]}) moves to ({v_transformed_coords[0]},{v_transformed_coords[1]})",
                             font="Comic Sans MS", font_size=24, color='LIGHTGREEN').to_corner(UR)
            self.play(ReplacementTransform(vect, vect_transformed), Write(result_tex), 
                    ReplacementTransform(exp_text, new_text))
            exp_text = new_text
            self.wait(1)
            self.play(FadeOut(vect_transformed), FadeOut(result_tex),  FadeOut(vector_tex))

        original_vectors_copy = [x.copy() for x in original_vectors]
        transformed_vectors_copy = [x.copy() for x in transformed_vectors]

        old_vector = original_vectors_copy
        new_vector = transformed_vectors_copy

        self.wait(1)
        self.play(*[Create(orig) for orig in original_vectors_copy])
        new_text = Text("All vectors 'transformed'", 
                        font="Comic Sans MS", 
                        font_size=24, 
                        color='LIGHTGREEN').to_corner(UR)
        

        self.play(
            *[ReplacementTransform(old, new) for old, new in zip(old_vector, new_vector)],
            ReplacementTransform(exp_text, new_text), run_time=2
        )
        exp_text = new_text
        old_vector = new_vector
        self.wait(1)

        dots = VGroup()
        step = 0.5 
        dot_radius = 0.05 

        dot_color = YELLOW 
        dot_opacity = 0.8  

        for x in np.arange(-5, 5 + step, step):
            for y in np.arange(-5, 5 + step, step):
                dot = Dot(point=[x, y, 0], radius=dot_radius, 
                          fill_color=dot_color, fill_opacity=dot_opacity)
                dots.add(dot)
        
        original_dots = dots.copy()
        transformed_dots = dots.copy()
        original_dots_copy = original_dots.copy()
        new_vector = [x.copy() for x in original_vectors]

        

        self.play(*[ReplacementTransform(old, new) for old, new in zip(old_vector, new_vector)])
        self.add(original_dots_copy)
        old_vector = new_vector

        self.wait(2)
        for dot, transformed_dot in zip(original_dots_copy, transformed_dots):
            x, y, _ = dot.get_center()
            new_coords = np.dot(matrix, [x, y])
            transformed_dot.move_to([*new_coords, 0])

        new_text = Text("The whole space is 'transformed'", 
                        font="Comic Sans MS", 
                        font_size=24, 
                        color='LIGHTGREEN').to_corner(UR)
        old_dot = original_dots_copy
        new_dot = transformed_dots.copy()
        new_vector = [x.copy() for x in transformed_vectors]

        self.play(ReplacementTransform(old_dot, new_dot),
                  ReplacementTransform(exp_text, new_text),
                  *[ReplacementTransform(old, new) for old, new in zip(old_vector, new_vector)],
                  run_time=2)
        
        self.wait(2)

        exp_text = new_text
        old_vector = new_vector
        old_dot = new_dot

        new_text = Text(f"Let's play this again.", 
                            font="Comic Sans MS", 
                             font_size=24, 
                             color='LIGHTGREEN').to_corner(UR)


        new_dot = original_dots.copy()
        new_vector = [x.copy() for x in original_vectors]
        self.play(ReplacementTransform(old_dot, new_dot),
                    *[ReplacementTransform(old, new) for old, new in zip(old_vector, new_vector)]
            ,ReplacementTransform(exp_text, new_text),run_time=2
        )
        self.wait(1)
        old_dot = new_dot
        old_vector = new_vector
        new_dot = transformed_dots.copy()
        new_vector = [x.copy() for x in transformed_vectors]

        self.play(ReplacementTransform(old_dot, new_dot),
                    *[ReplacementTransform(old, new) for old, new in zip(old_vector, new_vector)]
            ,ReplacementTransform(exp_text, new_text),run_time=2
        )

        self.wait(1)
        new_text = Text(f"Did you notice that some points don't get rotated with this matrix?", 
                    font="Comic Sans MS", 
                        font_size=24, 
                        color='ORANGE').to_corner(DR)
        self.play(Write(new_text))
        self.wait(2)

        old_vector = new_vector
        old_dot = new_dot

        new_dot = original_dots.copy()
        new_vector = [x.copy() for x in original_vectors]
        self.play(ReplacementTransform(old_dot, new_dot),
                    *[ReplacementTransform(old, new) for old, new in zip(old_vector, new_vector)], run_time=1
        )
        self.wait(1)
        old_dot = new_dot
        old_vector = new_vector
        new_dot = transformed_dots.copy()
        new_vector = [x.copy() for x in transformed_vectors]

        v_coords = np.array([1, -1])
        v_transformed_coords = np.dot(matrix, v_coords)
        vect = Arrow(start=ORIGIN, end=[*v_coords, 0], color=YELLOW, buff=0)
        vect_transformed = Arrow(start=ORIGIN, end=[*v_transformed_coords, 0], color=YELLOW, buff=0)
        v_orig = vect.copy()
        self.play(Create(vect))

        self.wait(1)
        self.play(ReplacementTransform(old_dot, new_dot),ReplacementTransform(vect, vect_transformed),
                    *[ReplacementTransform(old, new) for old, new in zip(old_vector, new_vector)]
            ,run_time=2
        )

        self.wait(1)
        new_text2 = Text(f"The yellow array is special for the matrix A\nMore on this in a few weeks :)", 
                    font="Comic Sans MS", 
                        font_size=24, 
                        color='ORANGE').to_corner(DR)
        self.play(FadeOut(new_text), Write(new_text2))
        self.wait(2)

class Matrix2(LinearTransformationScene):
    def __init__(self):
        LinearTransformationScene.__init__(
            self,
            show_coordinates=True,
            leave_ghost_vectors=True,
            show_basis_vectors=True,
        )

    def construct(self):

        matrix = [[1, 2], [2, 1]]

        matrix_tex = (
            MathTex("A = \\begin{bmatrix} 1 & 2 \\\ 2 & 1 \\end{bmatrix}")
            .to_edge(UL)
            .add_background_rectangle()
        )

        unit_square = self.get_unit_square()

        self.add_transformable_mobject(unit_square)
        self.add_background_mobject(matrix_tex)
        self.apply_matrix(matrix, run_time = 3)

        self.wait()

class Matrix1(LinearTransformationScene):
    def __init__(self):
        LinearTransformationScene.__init__(
            self,
            show_coordinates=True,
            leave_ghost_vectors=True,
            show_basis_vectors=True,
        )

    def construct(self):

        matrix = [[2, 1], [1, 2]]

        matrix_tex = (
            MathTex("A = \\begin{bmatrix} 2 & 1 \\\ 1 & 2 \\end{bmatrix}")
            .to_edge(UL)
            .add_background_rectangle()
        )

        unit_square = self.get_unit_square()

        self.add_transformable_mobject(unit_square)
        self.add_background_mobject(matrix_tex)
        self.apply_matrix(matrix, run_time = 3)

        self.wait()

class Matrix3(LinearTransformationScene):
    def __init__(self):
        LinearTransformationScene.__init__(
            self,
            show_coordinates=True,
            leave_ghost_vectors=True,
            show_basis_vectors=True,
        )

    def construct(self):

        matrix = [[1, 2], [2, 4]]

        matrix_tex = (
            MathTex("A = \\begin{bmatrix} 1 & 2 \\\ 2 & 4 \\end{bmatrix}")
            .to_edge(UL)
            .add_background_rectangle()
        )

        unit_square = self.get_unit_square()

        self.add_transformable_mobject(unit_square)
        self.add_background_mobject(matrix_tex)
        self.apply_matrix(matrix, run_time = 3)

        self.wait()


class MatrixDeterminant(Scene):


    def construct(self):

        def get_determinant_value(matrix_entries):
            matrix = np.array([
                [float(matrix_entries[0].get_tex_string()), float(matrix_entries[1].get_tex_string())],
                [float(matrix_entries[2].get_tex_string()), float(matrix_entries[3].get_tex_string())]
            ])
            return np.linalg.det(matrix)
        # Create title
        title = Text("Calculation for 2x2 Matrices")
        title.to_edge(UP)
        self.play(Write(title))

        # Adding matrix with white entries
        matrix = Matrix(
            [[1, 2],
             [-2, 3]]
        )
        matrix.next_to(title, DOWN, buff=1)
        self.play(Write(matrix))

        # Wait for a second
        self.wait(1)

        # Highlight the diagonal elements in green and off-diagonal elements in red
        matrix_entries = matrix.get_entries()
        self.play(
            matrix_entries[0].animate.set_color(GREEN),
            matrix_entries[3].animate.set_color(GREEN),
            matrix_entries[1].animate.set_color(RED),
            matrix_entries[2].animate.set_color(RED)
        )

         # Animate transitions of the matrix entries to the formula
        green_product = MathTex(f"{matrix_entries[0].get_tex_string()}", r"\times", f"{matrix_entries[3].get_tex_string()}", color=GREEN)
        red_product = MathTex(f"{matrix_entries[1].get_tex_string()}", r"\times", f"{matrix_entries[2].get_tex_string()}", color=RED)

        minus_text = MathTex(f") - (")
        close_par = MathTex(f")")

        # Position the products
        det_text = MathTex(f"det = (").next_to(matrix, 1.5*DOWN + LEFT) 
        green_product.next_to(det_text, RIGHT)
        minus_text.next_to(green_product, RIGHT)
        red_product.next_to(minus_text, RIGHT)
        close_par.next_to(red_product, RIGHT)
        
        # Transition animations
        self.play(Write(det_text), Write(minus_text), Write(close_par),
            Write(green_product),
            Write(red_product), run_time=1
        )
        self.play(
            ReplacementTransform(matrix_entries[0].copy(), green_product[0]),
            ReplacementTransform(matrix_entries[3].copy(), green_product[2]),
            ReplacementTransform(matrix_entries[1].copy(), red_product[0]),
            ReplacementTransform(matrix_entries[2].copy(), red_product[2]), run_time=3
        )

     # Keeping the scene on screen for a few more seconds
        self.wait(1)


        det_value = round(get_determinant_value(matrix_entries))  # round to get an integer value, adjust as needed
        final_determinant_value = MathTex(r" =", str(det_value)).next_to(close_par, RIGHT)


        self.play(Write(final_determinant_value), run_time=2)

        self.wait(3)


class MatrixDeterminant3(Scene):


    def construct(self):

        def get_determinant_value(matrix_entries):
            matrix = np.array([
                [float(matrix_entries[0].get_tex_string()), float(matrix_entries[1].get_tex_string())],
                [float(matrix_entries[2].get_tex_string()), float(matrix_entries[3].get_tex_string())]
            ])
            return np.linalg.det(matrix)

        def get_determinant_value3(matrix_entries):
            matrix = np.array([
                [float(matrix_entries[0].get_tex_string()), float(matrix_entries[1].get_tex_string()), float(matrix_entries[2].get_tex_string())],
                [float(matrix_entries[3].get_tex_string()), float(matrix_entries[4].get_tex_string()), float(matrix_entries[5].get_tex_string())],
                [float(matrix_entries[6].get_tex_string()), float(matrix_entries[7].get_tex_string()), float(matrix_entries[8].get_tex_string())]
            ])
            return np.linalg.det(matrix)
        # Create title
        title = Text("Calculation for 3x3 Matrices")
        title.to_edge(UP)
        self.play(Write(title))

        # Adding matrix with white entries
        matrix = Matrix(
            [[1, 2, 3],
             [-2, 3, 0],
            [1, 0, 2] ]
        )
        matrix.scale(0.75).next_to(title, DOWN, buff=1)
        self.play(Write(matrix))
        self.wait(1)


        matrix_entries = matrix.get_entries()
        self.play(matrix_entries[0].animate.set_color(GREEN))
        
        green_product = MathTex(f"{matrix_entries[0].get_tex_string()}", color=GREEN)
        green_product.next_to(matrix, 2.5*DOWN + 8*LEFT) 
        self.play(ReplacementTransform(matrix_entries[0].copy(), green_product))
        m1_values = [
            [matrix_entries[4].get_tex_string(), matrix_entries[5].get_tex_string()],
            [matrix_entries[7].get_tex_string(), matrix_entries[8].get_tex_string()]
        ]

        m1 = Matrix(m1_values,
            left_bracket="|",
            right_bracket="|").scale(0.75)
        m1.next_to(green_product)

        self.wait(1)


        minus_text = MathTex(f" - ", color=RED)
        minus_text.next_to(m1, RIGHT)

        red_product = MathTex(f"{matrix_entries[1].get_tex_string()}", color=RED)
        red_product.next_to(minus_text, RIGHT)
        self.play(Write(minus_text))
        self.play(ReplacementTransform(matrix_entries[1].copy(), red_product))
        self.play(matrix_entries[1].animate.set_color(RED))
        m2_values = [
            [matrix_entries[3].get_tex_string(), matrix_entries[5].get_tex_string()],
            [matrix_entries[6].get_tex_string(), matrix_entries[8].get_tex_string()]
        ]

        m2 = Matrix(m2_values,
            left_bracket="|",
            right_bracket="|").scale(0.75)
        m2.next_to(red_product)
        #self.play(Create(m2))
        self.wait(1)
        plus_text = MathTex(f" + ", color=GREEN)
        plus_text.next_to(m2, RIGHT)

        green_product2 = MathTex(f"{matrix_entries[2].get_tex_string()}", color=GREEN)
        green_product2.next_to(plus_text, RIGHT)
        self.play(Write(plus_text))
        self.play(ReplacementTransform(matrix_entries[2].copy(), green_product2))
        self.play(matrix_entries[2].animate.set_color(GREEN))
        m3_values = [
            [matrix_entries[3].get_tex_string(), matrix_entries[4].get_tex_string()],
            [matrix_entries[6].get_tex_string(), matrix_entries[7].get_tex_string()]
        ]

        m3 = Matrix(m3_values,
            left_bracket="|",
            right_bracket="|").scale(0.75)
        m3.next_to(green_product2)

        self.wait(1)

        highlight_color = GREEN
        first_row = VGroup(matrix.get_rows()[0])
        first_column = VGroup(matrix.get_columns()[0])
        rect1 = SurroundingRectangle(first_row, color=highlight_color)
        rect2 = SurroundingRectangle(first_column, color=highlight_color)
        self.play(Create(rect1), Create(rect2))
        self.wait(1)
        self.play(
            matrix_entries[4].animate.set_color(YELLOW),
            matrix_entries[5].animate.set_color(YELLOW),
            matrix_entries[7].animate.set_color(YELLOW),
            matrix_entries[8].animate.set_color(YELLOW)
        )

        self.play(
            ReplacementTransform(matrix_entries[4].copy(), m1.get_entries()[0]),
            ReplacementTransform(matrix_entries[5].copy(), m1.get_entries()[1]),
            ReplacementTransform(matrix_entries[7].copy(), m1.get_entries()[2]),
            ReplacementTransform(matrix_entries[8].copy(), m1.get_entries()[3]),
            *[FadeIn(mobj) for mobj in m1.get_brackets()]
        )

        self.play(
            matrix_entries[4].animate.set_color(WHITE),
            matrix_entries[5].animate.set_color(WHITE),
            matrix_entries[7].animate.set_color(WHITE),
            matrix_entries[8].animate.set_color(WHITE), FadeOut(rect1), FadeOut(rect2)
        )
        self.wait(1)

        highlight_color = RED
        first_row = VGroup(matrix.get_rows()[0])
        second_column = VGroup(matrix.get_columns()[1])
        rect1 = SurroundingRectangle(first_row, color=highlight_color)
        rect2 = SurroundingRectangle(second_column, color=highlight_color)
        self.play(Create(rect1), Create(rect2))
        self.wait(1)
        self.play(
            matrix_entries[3].animate.set_color(YELLOW),
            matrix_entries[5].animate.set_color(YELLOW),
            matrix_entries[6].animate.set_color(YELLOW),
            matrix_entries[8].animate.set_color(YELLOW),
            *[FadeIn(mobj) for mobj in m2.get_brackets()] 
        )

        self.play(
            ReplacementTransform(matrix_entries[3].copy(), m2.get_entries()[0]),
            ReplacementTransform(matrix_entries[5].copy(), m2.get_entries()[1]),
            ReplacementTransform(matrix_entries[6].copy(), m2.get_entries()[2]),
            ReplacementTransform(matrix_entries[8].copy(), m2.get_entries()[3])
        )

        self.play(
            matrix_entries[3].animate.set_color(WHITE),
            matrix_entries[5].animate.set_color(WHITE),
            matrix_entries[6].animate.set_color(WHITE),
            matrix_entries[8].animate.set_color(WHITE), FadeOut(rect1), FadeOut(rect2)
        )


        highlight_color = GREEN
        first_row = VGroup(matrix.get_rows()[0])
        third_column = VGroup(matrix.get_columns()[2])
        rect1 = SurroundingRectangle(first_row, color=highlight_color)
        rect2 = SurroundingRectangle(third_column, color=highlight_color)
        self.play(Create(rect1), Create(rect2))
        self.wait(1)
        self.play(
            matrix_entries[3].animate.set_color(YELLOW),
            matrix_entries[4].animate.set_color(YELLOW),
            matrix_entries[6].animate.set_color(YELLOW),
            matrix_entries[7].animate.set_color(YELLOW),
            *[FadeIn(mobj) for mobj in m3.get_brackets()] 
        )

        self.play(
            ReplacementTransform(matrix_entries[3].copy(), m3.get_entries()[0]),
            ReplacementTransform(matrix_entries[4].copy(), m3.get_entries()[1]),
            ReplacementTransform(matrix_entries[6].copy(), m3.get_entries()[2]),
            ReplacementTransform(matrix_entries[7].copy(), m3.get_entries()[3])
        )

        self.play(
            matrix_entries[3].animate.set_color(WHITE),
            matrix_entries[4].animate.set_color(WHITE),
            matrix_entries[6].animate.set_color(WHITE),
            matrix_entries[7].animate.set_color(WHITE), FadeOut(rect1), FadeOut(rect2)
        )


        self.wait(1)
        m1_entries = m1.get_entries()
        green_product_copy = MathTex(f"{matrix_entries[0].get_tex_string()}", color=GREEN)
        green_product_copy.next_to(green_product, 3*DOWN + 1.5*LEFT)
        self.play(ReplacementTransform(green_product.copy(), green_product_copy))

        prod1 = f"{m1_entries[0].get_tex_string()}"
        prod2 = f"{m1_entries[3].get_tex_string()}"
        prod3 = f"{m1_entries[1].get_tex_string()}"
        prod4 = f"{m1_entries[2].get_tex_string()}"
        replacement_text1 = MathTex(fr"({prod1}\times{prod2} - {prod3}\times{prod4})").scale(0.75).next_to(green_product_copy, RIGHT)
        self.play(Write(replacement_text1))

        minus_text_copy = MathTex(f" - ", color=RED).next_to(replacement_text1, RIGHT)
        red_product_copy = MathTex(f"{matrix_entries[1].get_tex_string()}", color=RED).next_to(minus_text_copy)
        self.play(ReplacementTransform(minus_text.copy(), minus_text_copy), ReplacementTransform(red_product.copy(), red_product_copy))
        m2_entries = m2.get_entries()
        prod1 = f"{m2_entries[0].get_tex_string()}"
        prod2 = f"{m2_entries[3].get_tex_string()}"
        prod3 = f"{m2_entries[1].get_tex_string()}"
        prod4 = f"{m2_entries[2].get_tex_string()}"
        replacement_text2 = MathTex(fr"({prod1}\times{prod2} - {prod3}\times{prod4})").scale(0.75).next_to(red_product_copy, RIGHT)
        self.play(Write(replacement_text2))

        plus_text_copy = MathTex(f" + ", color=GREEN).next_to(replacement_text2, RIGHT)
        green_product2_copy = MathTex(f"{matrix_entries[2].get_tex_string()}", color=GREEN).next_to(plus_text_copy)
        self.play(ReplacementTransform(plus_text.copy(), plus_text_copy), ReplacementTransform(green_product2.copy(), green_product2_copy))
        m3_entries = m3.get_entries()
        prod1 = f"{m3_entries[0].get_tex_string()}"
        prod2 = f"{m3_entries[3].get_tex_string()}"
        prod3 = f"{m3_entries[1].get_tex_string()}"
        prod4 = f"{m3_entries[2].get_tex_string()}"
        replacement_text3 = MathTex(fr"({prod1}\times{prod2} - {prod3}\times{prod4})").scale(0.75).next_to(green_product2_copy, RIGHT)
        self.play(Write(replacement_text3))
        self.wait(1)

        det_value1 = round(get_determinant_value(m1_entries))  # round to get an integer value, adjust as needed
        det1 = MathTex(fr"\times{det_value1}").next_to(green_product_copy, RIGHT)
        self.play(ReplacementTransform(replacement_text1, det1), run_time = 0.5)
        det_value2 = round(get_determinant_value(m2_entries))  # round to get an integer value, adjust as needed
        det2 = MathTex(fr"\times{det_value2}").next_to(red_product_copy, RIGHT)
        self.play(ReplacementTransform(replacement_text2, det2), run_time = 0.5)
        det_value3 = round(get_determinant_value(m3_entries))  # round to get an integer value, adjust as needed
        det3 = MathTex(fr"\times{det_value3}").next_to(green_product2_copy, RIGHT)
        self.play(ReplacementTransform(replacement_text3, det3), run_time = 0.5)

        final_det = round(get_determinant_value3(matrix_entries))
        final_determinant_value = MathTex(fr"det(A) = {final_det}").next_to(red_product_copy, DOWN*2)
        self.play(Write(final_determinant_value), run_time=2)

        self.wait(3)



class MatrixDeterminant4(Scene):


    def construct(self):

        def get_determinant_value(matrix_entries):
            matrix = np.array([
                [float(matrix_entries[0].get_tex_string()), float(matrix_entries[1].get_tex_string())],
                [float(matrix_entries[2].get_tex_string()), float(matrix_entries[3].get_tex_string())]
            ])
            return np.linalg.det(matrix)

        def get_determinant_value3(matrix_entries):
            matrix = np.array([
                [float(matrix_entries[0].get_tex_string()), float(matrix_entries[1].get_tex_string()), float(matrix_entries[2].get_tex_string())],
                [float(matrix_entries[3].get_tex_string()), float(matrix_entries[4].get_tex_string()), float(matrix_entries[5].get_tex_string())],
                [float(matrix_entries[6].get_tex_string()), float(matrix_entries[7].get_tex_string()), float(matrix_entries[8].get_tex_string())]
            ])
            return np.linalg.det(matrix)
        def get_determinant_value4(matrix_entries):
            matrix = np.array([
                [float(matrix_entries[0].get_tex_string()), float(matrix_entries[1].get_tex_string()), float(matrix_entries[2].get_tex_string()), float(matrix_entries[3].get_tex_string())],
                [float(matrix_entries[4].get_tex_string()), float(matrix_entries[5].get_tex_string()),float(matrix_entries[6].get_tex_string()), float(matrix_entries[7].get_tex_string())],
                [float(matrix_entries[8].get_tex_string()), float(matrix_entries[9].get_tex_string()),float(matrix_entries[10].get_tex_string()), float(matrix_entries[11].get_tex_string())],
                [float(matrix_entries[12].get_tex_string()), float(matrix_entries[13].get_tex_string()),float(matrix_entries[14].get_tex_string()), float(matrix_entries[15].get_tex_string())]
            ])
            return np.linalg.det(matrix)
        def determinant_of_3(matrix, coeff, is_last=False):
            matrix_entries = matrix.get_entries()
            green_product = MathTex(f"{matrix_entries[0].get_tex_string()}", color=GREEN)
            if is_last:
                green_product.next_to(matrix, 2.5*DOWN + 10*LEFT)
                run_time = 0.05
            else:
                green_product.next_to(matrix, 2.5*DOWN)
                run_time = 0.1
            self.play(matrix_entries[0].animate.set_color(GREEN), run_time=run_time)  
            self.play(ReplacementTransform(matrix_entries[0].copy(), green_product), run_time=run_time)
            m1_values = [
                [matrix_entries[4].get_tex_string(), matrix_entries[5].get_tex_string()],
                [matrix_entries[7].get_tex_string(), matrix_entries[8].get_tex_string()]
            ]

            m1 = Matrix(m1_values,
                left_bracket="|",
                right_bracket="|").scale(0.75)
            m1.next_to(green_product)

            minus_text = MathTex(f" - ", color=RED)
            minus_text.next_to(m1, RIGHT)

            red_product = MathTex(f"{matrix_entries[1].get_tex_string()}", color=RED)
            red_product.next_to(minus_text, RIGHT)
            self.play(Write(minus_text), run_time=run_time)
            self.play(ReplacementTransform(matrix_entries[1].copy(), red_product), run_time=run_time)
            self.play(matrix_entries[1].animate.set_color(RED), run_time=run_time)
            m2_values = [
                [matrix_entries[3].get_tex_string(), matrix_entries[5].get_tex_string()],
                [matrix_entries[6].get_tex_string(), matrix_entries[8].get_tex_string()]
            ]

            m2 = Matrix(m2_values,
                left_bracket="|",
                right_bracket="|").scale(0.75)
            m2.next_to(red_product)
            plus_text = MathTex(f" + ", color=GREEN)
            plus_text.next_to(m2, RIGHT)

            green_product2 = MathTex(f"{matrix_entries[2].get_tex_string()}", color=GREEN)
            green_product2.next_to(plus_text, RIGHT)
            self.play(Write(plus_text), run_time=run_time)
            self.play(ReplacementTransform(matrix_entries[2].copy(), green_product2), run_time=run_time)
            self.play(matrix_entries[2].animate.set_color(GREEN), run_time=run_time)
            m3_values = [
                [matrix_entries[3].get_tex_string(), matrix_entries[4].get_tex_string()],
                [matrix_entries[6].get_tex_string(), matrix_entries[7].get_tex_string()]
            ]

            m3 = Matrix(m3_values,
                left_bracket="|",
                right_bracket="|").scale(0.75)
            m3.next_to(green_product2)

            highlight_color = GREEN
            first_row = VGroup(matrix.get_rows()[0])
            first_column = VGroup(matrix.get_columns()[0])
            rect1 = SurroundingRectangle(first_row, color=highlight_color)
            rect2 = SurroundingRectangle(first_column, color=highlight_color)
            self.play(Create(rect1), Create(rect2), run_time=run_time)
            self.play(
                matrix_entries[4].animate.set_color(YELLOW),
                matrix_entries[5].animate.set_color(YELLOW),
                matrix_entries[7].animate.set_color(YELLOW),
                matrix_entries[8].animate.set_color(YELLOW), run_time=run_time
            )

            self.play(
                ReplacementTransform(matrix_entries[4].copy(), m1.get_entries()[0]),
                ReplacementTransform(matrix_entries[5].copy(), m1.get_entries()[1]),
                ReplacementTransform(matrix_entries[7].copy(), m1.get_entries()[2]),
                ReplacementTransform(matrix_entries[8].copy(), m1.get_entries()[3]),
                *[FadeIn(mobj) for mobj in m1.get_brackets()], run_time=run_time
            )

            self.play(
                matrix_entries[4].animate.set_color(WHITE),
                matrix_entries[5].animate.set_color(WHITE),
                matrix_entries[7].animate.set_color(WHITE),
                matrix_entries[8].animate.set_color(WHITE), FadeOut(rect1), FadeOut(rect2), run_time=run_time
            )

            highlight_color = RED
            first_row = VGroup(matrix.get_rows()[0])
            second_column = VGroup(matrix.get_columns()[1])
            rect1 = SurroundingRectangle(first_row, color=highlight_color)
            rect2 = SurroundingRectangle(second_column, color=highlight_color)
            self.play(Create(rect1), Create(rect2), run_time=run_time)
            self.play(
                matrix_entries[3].animate.set_color(YELLOW),
                matrix_entries[5].animate.set_color(YELLOW),
                matrix_entries[6].animate.set_color(YELLOW),
                matrix_entries[8].animate.set_color(YELLOW),
                *[FadeIn(mobj) for mobj in m2.get_brackets()] , run_time=run_time
            )

            self.play(
                ReplacementTransform(matrix_entries[3].copy(), m2.get_entries()[0]),
                ReplacementTransform(matrix_entries[5].copy(), m2.get_entries()[1]),
                ReplacementTransform(matrix_entries[6].copy(), m2.get_entries()[2]),
                ReplacementTransform(matrix_entries[8].copy(), m2.get_entries()[3]), run_time=run_time
            )

            self.play(
                matrix_entries[3].animate.set_color(WHITE),
                matrix_entries[5].animate.set_color(WHITE),
                matrix_entries[6].animate.set_color(WHITE),
                matrix_entries[8].animate.set_color(WHITE), FadeOut(rect1), FadeOut(rect2), run_time=run_time
            )


            highlight_color = GREEN
            first_row = VGroup(matrix.get_rows()[0])
            third_column = VGroup(matrix.get_columns()[2])
            rect1 = SurroundingRectangle(first_row, color=highlight_color)
            rect2 = SurroundingRectangle(third_column, color=highlight_color)
            self.play(Create(rect1), Create(rect2), run_time=0.1)
            self.play(
                matrix_entries[3].animate.set_color(YELLOW),
                matrix_entries[4].animate.set_color(YELLOW),
                matrix_entries[6].animate.set_color(YELLOW),
                matrix_entries[7].animate.set_color(YELLOW),
                *[FadeIn(mobj) for mobj in m3.get_brackets()] , run_time=run_time
            )

            self.play(
                ReplacementTransform(matrix_entries[3].copy(), m3.get_entries()[0]),
                ReplacementTransform(matrix_entries[4].copy(), m3.get_entries()[1]),
                ReplacementTransform(matrix_entries[6].copy(), m3.get_entries()[2]),
                ReplacementTransform(matrix_entries[7].copy(), m3.get_entries()[3]), run_time=run_time
            )

            self.play(
                matrix_entries[3].animate.set_color(WHITE),
                matrix_entries[4].animate.set_color(WHITE),
                matrix_entries[6].animate.set_color(WHITE),
                matrix_entries[7].animate.set_color(WHITE), FadeOut(rect1), FadeOut(rect2), run_time=run_time
            )

            m1_entries = m1.get_entries()
            green_product_copy = MathTex(f"{matrix_entries[0].get_tex_string()}", color=GREEN)
            green_product_copy.next_to(green_product, 3*DOWN + 1.5*LEFT)
            self.play(ReplacementTransform(green_product.copy(), green_product_copy), run_time=run_time)

            prod1 = f"{m1_entries[0].get_tex_string()}"
            prod2 = f"{m1_entries[3].get_tex_string()}"
            prod3 = f"{m1_entries[1].get_tex_string()}"
            prod4 = f"{m1_entries[2].get_tex_string()}"
            replacement_text1 = MathTex(fr"({prod1}\times{prod2} - {prod3}\times{prod4})").scale(0.75).next_to(green_product_copy, RIGHT)
            self.play(Write(replacement_text1), run_time=run_time)

            minus_text_copy = MathTex(f" - ", color=RED).next_to(replacement_text1, RIGHT)
            red_product_copy = MathTex(f"{matrix_entries[1].get_tex_string()}", color=RED).next_to(minus_text_copy)
            self.play(ReplacementTransform(minus_text.copy(), minus_text_copy), ReplacementTransform(red_product.copy(), red_product_copy), run_time=run_time)
            m2_entries = m2.get_entries()
            prod1 = f"{m2_entries[0].get_tex_string()}"
            prod2 = f"{m2_entries[3].get_tex_string()}"
            prod3 = f"{m2_entries[1].get_tex_string()}"
            prod4 = f"{m2_entries[2].get_tex_string()}"
            replacement_text2 = MathTex(fr"({prod1}\times{prod2} - {prod3}\times{prod4})").scale(0.75).next_to(red_product_copy, RIGHT)
            self.play(Write(replacement_text2), run_time=run_time)

            plus_text_copy = MathTex(f" + ", color=GREEN).next_to(replacement_text2, RIGHT)
            green_product2_copy = MathTex(f"{matrix_entries[2].get_tex_string()}", color=GREEN).next_to(plus_text_copy)
            self.play(ReplacementTransform(plus_text.copy(), plus_text_copy), ReplacementTransform(green_product2.copy(), green_product2_copy), run_time=run_time)
            m3_entries = m3.get_entries()
            prod1 = f"{m3_entries[0].get_tex_string()}"
            prod2 = f"{m3_entries[3].get_tex_string()}"
            prod3 = f"{m3_entries[1].get_tex_string()}"
            prod4 = f"{m3_entries[2].get_tex_string()}"
            replacement_text3 = MathTex(fr"({prod1}\times{prod2} - {prod3}\times{prod4})").scale(0.75).next_to(green_product2_copy, RIGHT)
            self.play(Write(replacement_text3), run_time=run_time)
            det_value1 = round(get_determinant_value(m1_entries))  # round to get an integer value, adjust as needed
            det1 = MathTex(fr"\times{det_value1}").next_to(green_product_copy, RIGHT)
            self.play(ReplacementTransform(replacement_text1, det1), run_time=run_time)
            det_value2 = round(get_determinant_value(m2_entries))  # round to get an integer value, adjust as needed
            det2 = MathTex(fr"\times{det_value2}").next_to(red_product_copy, RIGHT)
            self.play(ReplacementTransform(replacement_text2, det2), run_time=run_time)
            det_value3 = round(get_determinant_value(m3_entries))  # round to get an integer value, adjust as needed
            det3 = MathTex(fr"\times{det_value3}").next_to(green_product2_copy, RIGHT)
            self.play(ReplacementTransform(replacement_text3, det3), run_time=run_time)

            final_det = round(get_determinant_value3(matrix_entries))
            final_determinant_value = MathTex(fr"\times {final_det}").next_to(coeff, RIGHT)
            self.play(ReplacementTransform(matrix, final_determinant_value), run_time=run_time*5)
            self.play(FadeOut(m1), FadeOut(m2), FadeOut(m3),
                      FadeOut(det1), FadeOut(det2), FadeOut(det3),
                    FadeOut(green_product), FadeOut(red_product), FadeOut(green_product2),
                      FadeOut(plus_text), FadeOut(minus_text), FadeOut(minus2_text), 
                      FadeOut(plus_text_copy), FadeOut(minus_text_copy), 
                      FadeOut(green_product_copy), FadeOut(red_product_copy), FadeOut(green_product2_copy), 
                      run_time=run_time)
            


        title = Text("Calculation for 4x4 Matrices")
        title.to_edge(UP)
        self.play(Write(title))

        # Adding matrix with white entries
        matrix = Matrix(
            [[1, 2, 3, 4],
            [-2, 1, 0, 1],
            [1, 0, 2, -3],
            [3, 2, 3, 1] ]
        )
        matrix.scale(0.65).next_to(title, 0.5*DOWN, buff=1)
        self.play(Write(matrix))
  
        matrix_entries = matrix.get_entries()
        self.play(matrix_entries[0].animate.set_color(GREEN),
                  matrix_entries[1].animate.set_color(RED),
                  matrix_entries[2].animate.set_color(GREEN),
                  matrix_entries[3].animate.set_color(RED)
                  )
    
        green_product = MathTex(f"{matrix_entries[0].get_tex_string()}", color=GREEN)
        green_product.to_edge(LEFT).shift(2*DOWN) 
        self.play(ReplacementTransform(matrix_entries[0].copy(), green_product), run_time=0.5)
        m1_values = [
            [matrix_entries[5].get_tex_string(), matrix_entries[6].get_tex_string(), matrix_entries[7].get_tex_string()],
            [matrix_entries[9].get_tex_string(), matrix_entries[10].get_tex_string(), matrix_entries[11].get_tex_string()],
            [matrix_entries[13].get_tex_string(), matrix_entries[14].get_tex_string(), matrix_entries[15].get_tex_string()],
        ]

        m1 = Matrix(m1_values,
            left_bracket="|",
            right_bracket="|").scale(0.65)
        m1.next_to(green_product)

        minus_text = MathTex(f" - ", color=RED)
        minus_text.next_to(m1, RIGHT)

        red_product = MathTex(f"{matrix_entries[1].get_tex_string()}", color=RED)
        red_product.next_to(minus_text, RIGHT)
        self.play(Write(minus_text), run_time=0.5)
        self.play(ReplacementTransform(matrix_entries[1].copy(), red_product), run_time=0.5)
        m2_values = [
            [matrix_entries[4].get_tex_string(), matrix_entries[6].get_tex_string(), matrix_entries[7].get_tex_string()],
            [matrix_entries[8].get_tex_string(), matrix_entries[10].get_tex_string(), matrix_entries[11].get_tex_string()],
            [matrix_entries[12].get_tex_string(), matrix_entries[14].get_tex_string(), matrix_entries[15].get_tex_string()],
        ]

        m2 = Matrix(m2_values,
            left_bracket="|",
            right_bracket="|").scale(0.65)
        m2.next_to(red_product)
        plus_text = MathTex(f" + ", color=GREEN)
        plus_text.next_to(m2, RIGHT)

        green_product2 = MathTex(f"{matrix_entries[2].get_tex_string()}", color=GREEN)
        green_product2.next_to(plus_text, RIGHT)
        self.play(Write(plus_text), run_time=0.5)
        self.play(ReplacementTransform(matrix_entries[2].copy(), green_product2), run_time=0.5)
        m3_values = [
            [matrix_entries[4].get_tex_string(), matrix_entries[5].get_tex_string(), matrix_entries[7].get_tex_string()],
            [matrix_entries[8].get_tex_string(), matrix_entries[9].get_tex_string(), matrix_entries[11].get_tex_string()],
            [matrix_entries[12].get_tex_string(), matrix_entries[13].get_tex_string(), matrix_entries[15].get_tex_string()],
        ]
        m3 = Matrix(m3_values,
            left_bracket="|",
            right_bracket="|").scale(0.65)
        m3.next_to(green_product2)

        minus2_text = MathTex(f" - ", color=RED).next_to(m3, RIGHT)

        red_product2 = MathTex(f"{matrix_entries[3].get_tex_string()}", color=RED)
        red_product2.next_to(minus2_text, RIGHT)
        self.play(Write(minus2_text), run_time=0.5)
        self.play(ReplacementTransform(matrix_entries[3].copy(), red_product2), run_time=0.5)
        m4_values = [
            [matrix_entries[4].get_tex_string(), matrix_entries[5].get_tex_string(), matrix_entries[6].get_tex_string()],
            [matrix_entries[8].get_tex_string(), matrix_entries[9].get_tex_string(), matrix_entries[10].get_tex_string()],
            [matrix_entries[12].get_tex_string(), matrix_entries[13].get_tex_string(), matrix_entries[14].get_tex_string()],
        ]
        m4 = Matrix(m4_values,
            left_bracket="|",
            right_bracket="|").scale(0.65)
        m4.next_to(red_product2)

        self.wait(1)

        highlight_color = GREEN
        first_row = VGroup(matrix.get_rows()[0])
        first_column = VGroup(matrix.get_columns()[0])
        rect1 = SurroundingRectangle(first_row, color=highlight_color)
        rect2 = SurroundingRectangle(first_column, color=highlight_color)
        self.play(Create(rect1), Create(rect2))
        self.wait(1)
        self.play(
            matrix_entries[5].animate.set_color(YELLOW),
            matrix_entries[6].animate.set_color(YELLOW),
            matrix_entries[7].animate.set_color(YELLOW),
            matrix_entries[9].animate.set_color(YELLOW),
            matrix_entries[10].animate.set_color(YELLOW),
            matrix_entries[11].animate.set_color(YELLOW),
            matrix_entries[13].animate.set_color(YELLOW),
            matrix_entries[14].animate.set_color(YELLOW),
            matrix_entries[15].animate.set_color(YELLOW)
        )

        self.play(
            ReplacementTransform(matrix_entries[5].copy(), m1.get_entries()[0]),
            ReplacementTransform(matrix_entries[6].copy(), m1.get_entries()[1]),
            ReplacementTransform(matrix_entries[7].copy(), m1.get_entries()[2]),
            ReplacementTransform(matrix_entries[9].copy(), m1.get_entries()[3]),
            ReplacementTransform(matrix_entries[10].copy(), m1.get_entries()[4]),
            ReplacementTransform(matrix_entries[11].copy(), m1.get_entries()[5]),
            ReplacementTransform(matrix_entries[13].copy(), m1.get_entries()[6]),
            ReplacementTransform(matrix_entries[14].copy(), m1.get_entries()[7]),
            ReplacementTransform(matrix_entries[15].copy(), m1.get_entries()[8]),
            *[FadeIn(mobj) for mobj in m1.get_brackets()]
        )

        self.play(
            matrix_entries[5].animate.set_color(WHITE),
            matrix_entries[6].animate.set_color(WHITE),
            matrix_entries[7].animate.set_color(WHITE),
            matrix_entries[9].animate.set_color(WHITE),
            matrix_entries[10].animate.set_color(WHITE),
            matrix_entries[11].animate.set_color(WHITE),
            matrix_entries[13].animate.set_color(WHITE),
            matrix_entries[14].animate.set_color(WHITE),
            matrix_entries[15].animate.set_color(WHITE), FadeOut(rect1), FadeOut(rect2)
        )

        highlight_color = RED
        first_row = VGroup(matrix.get_rows()[0])
        second_column = VGroup(matrix.get_columns()[1])
        rect1 = SurroundingRectangle(first_row, color=highlight_color)
        rect2 = SurroundingRectangle(second_column, color=highlight_color)
        self.play(Create(rect1), Create(rect2))
        self.play(
            matrix_entries[4].animate.set_color(YELLOW),
            matrix_entries[6].animate.set_color(YELLOW),
            matrix_entries[7].animate.set_color(YELLOW),
            matrix_entries[8].animate.set_color(YELLOW),
            matrix_entries[10].animate.set_color(YELLOW),
            matrix_entries[11].animate.set_color(YELLOW),
            matrix_entries[12].animate.set_color(YELLOW),
            matrix_entries[14].animate.set_color(YELLOW),
            matrix_entries[15].animate.set_color(YELLOW)
        )
        self.play(
            ReplacementTransform(matrix_entries[4].copy(), m2.get_entries()[0]),
            ReplacementTransform(matrix_entries[6].copy(), m2.get_entries()[1]),
            ReplacementTransform(matrix_entries[7].copy(), m2.get_entries()[2]),
            ReplacementTransform(matrix_entries[8].copy(), m2.get_entries()[3]),
            ReplacementTransform(matrix_entries[10].copy(), m2.get_entries()[4]),
            ReplacementTransform(matrix_entries[11].copy(), m2.get_entries()[5]),
            ReplacementTransform(matrix_entries[12].copy(), m2.get_entries()[6]),
            ReplacementTransform(matrix_entries[14].copy(), m2.get_entries()[7]),
            ReplacementTransform(matrix_entries[15].copy(), m2.get_entries()[8]),
            *[FadeIn(mobj) for mobj in m2.get_brackets()] 
        )



        self.play(
            matrix_entries[4].animate.set_color(WHITE),
            matrix_entries[6].animate.set_color(WHITE),
            matrix_entries[7].animate.set_color(WHITE),
            matrix_entries[8].animate.set_color(WHITE),
            matrix_entries[10].animate.set_color(WHITE),
            matrix_entries[11].animate.set_color(WHITE),
            matrix_entries[12].animate.set_color(WHITE),
            matrix_entries[14].animate.set_color(WHITE),
            matrix_entries[15].animate.set_color(WHITE), FadeOut(rect1), FadeOut(rect2)
        )


        highlight_color = GREEN
        first_row = VGroup(matrix.get_rows()[0])
        third_column = VGroup(matrix.get_columns()[2])
        rect1 = SurroundingRectangle(first_row, color=highlight_color)
        rect2 = SurroundingRectangle(third_column, color=highlight_color)
        self.play(Create(rect1), Create(rect2), run_time=0.5)
        self.play(
            matrix_entries[4].animate.set_color(YELLOW),
            matrix_entries[5].animate.set_color(YELLOW),
            matrix_entries[7].animate.set_color(YELLOW),
            matrix_entries[8].animate.set_color(YELLOW),
            matrix_entries[9].animate.set_color(YELLOW),
            matrix_entries[11].animate.set_color(YELLOW),
            matrix_entries[12].animate.set_color(YELLOW),
            matrix_entries[13].animate.set_color(YELLOW),
            matrix_entries[15].animate.set_color(YELLOW),
            *[FadeIn(mobj) for mobj in m3.get_brackets()] , run_time=0.5
        )

        self.play(
            ReplacementTransform(matrix_entries[4].copy(), m3.get_entries()[0]),
            ReplacementTransform(matrix_entries[5].copy(), m3.get_entries()[1]),
            ReplacementTransform(matrix_entries[7].copy(), m3.get_entries()[2]),
            ReplacementTransform(matrix_entries[8].copy(), m3.get_entries()[3]),
            ReplacementTransform(matrix_entries[9].copy(), m3.get_entries()[4]),
            ReplacementTransform(matrix_entries[11].copy(), m3.get_entries()[5]),
            ReplacementTransform(matrix_entries[12].copy(), m3.get_entries()[6]),
            ReplacementTransform(matrix_entries[13].copy(), m3.get_entries()[7]),
            ReplacementTransform(matrix_entries[15].copy(), m3.get_entries()[8]),run_time=0.5
        )

        self.play(
            matrix_entries[4].animate.set_color(WHITE),
            matrix_entries[5].animate.set_color(WHITE),
            matrix_entries[7].animate.set_color(WHITE),
            matrix_entries[8].animate.set_color(WHITE),
            matrix_entries[9].animate.set_color(WHITE),
            matrix_entries[11].animate.set_color(WHITE),
            matrix_entries[12].animate.set_color(WHITE),
            matrix_entries[13].animate.set_color(WHITE),
            matrix_entries[15].animate.set_color(WHITE),
            FadeOut(rect1), FadeOut(rect2), run_time=0.5
        )


        highlight_color = RED
        first_row = VGroup(matrix.get_rows()[0])
        fourth_column = VGroup(matrix.get_columns()[3])
        rect1 = SurroundingRectangle(first_row, color=highlight_color)
        rect2 = SurroundingRectangle(fourth_column, color=highlight_color)
        self.play(Create(rect1), Create(rect2), run_time=0.5)
        self.play(
            matrix_entries[4].animate.set_color(YELLOW),
            matrix_entries[5].animate.set_color(YELLOW),
            matrix_entries[6].animate.set_color(YELLOW),
            matrix_entries[8].animate.set_color(YELLOW),
            matrix_entries[9].animate.set_color(YELLOW),
            matrix_entries[10].animate.set_color(YELLOW),
            matrix_entries[12].animate.set_color(YELLOW),
            matrix_entries[13].animate.set_color(YELLOW),
            matrix_entries[14].animate.set_color(YELLOW),
            *[FadeIn(mobj) for mobj in m4.get_brackets()] , run_time=0.5
        )

        self.play(
            ReplacementTransform(matrix_entries[4].copy(), m4.get_entries()[0]),
            ReplacementTransform(matrix_entries[5].copy(), m4.get_entries()[1]),
            ReplacementTransform(matrix_entries[6].copy(), m4.get_entries()[2]),
            ReplacementTransform(matrix_entries[8].copy(), m4.get_entries()[3]),
            ReplacementTransform(matrix_entries[9].copy(), m4.get_entries()[4]),
            ReplacementTransform(matrix_entries[10].copy(), m4.get_entries()[5]),
            ReplacementTransform(matrix_entries[12].copy(), m4.get_entries()[6]),
            ReplacementTransform(matrix_entries[13].copy(), m4.get_entries()[7]),
            ReplacementTransform(matrix_entries[14].copy(), m4.get_entries()[8]), run_time=0.5

        )

        self.play(
            matrix_entries[4].animate.set_color(WHITE),
            matrix_entries[5].animate.set_color(WHITE),
            matrix_entries[6].animate.set_color(WHITE),
            matrix_entries[8].animate.set_color(WHITE),
            matrix_entries[9].animate.set_color(WHITE),
            matrix_entries[10].animate.set_color(WHITE),
            matrix_entries[12].animate.set_color(WHITE),
            matrix_entries[13].animate.set_color(WHITE),
            matrix_entries[14].animate.set_color(WHITE),
              FadeOut(rect1), FadeOut(rect2), run_time=0.5
        )
        self.play(FadeOut(matrix))
        self.play(m1.animate.shift(UP*4), m2.animate.shift(UP*4), m3.animate.shift(UP*4), m4.animate.shift(UP*4),
                  green_product.animate.shift(UP*4), green_product2.animate.shift(UP*4), red_product.animate.shift(UP*4), 
                  red_product2.animate.shift(UP*4), minus_text.animate.shift(UP*4), minus2_text.animate.shift(UP*4),
                    plus_text.animate.shift(UP*4))
        determinant_of_3(m1, green_product)
        determinant_of_3(m2, red_product)
        determinant_of_3(m3, green_product2, is_last=True)
        determinant_of_3(m4, red_product2, is_last=True)
        final_det = round(get_determinant_value4(matrix_entries))
        final_determinant_value = MathTex(fr"det =  {final_det}").shift(2*DOWN)
        self.play(Write(final_determinant_value))
        self.wait(3)


class MatrixInverse(Scene):


    def construct(self):

        def get_determinant_value(matrix_entries):
            matrix = np.array([
                [float(matrix_entries[0].get_tex_string()), float(matrix_entries[1].get_tex_string())],
                [float(matrix_entries[2].get_tex_string()), float(matrix_entries[3].get_tex_string())]
            ])
            return np.linalg.det(matrix)

        def get_determinant_value3(matrix_entries):
            matrix = np.array([
                [float(matrix_entries[0].get_tex_string()), float(matrix_entries[1].get_tex_string()), float(matrix_entries[2].get_tex_string())],
                [float(matrix_entries[3].get_tex_string()), float(matrix_entries[4].get_tex_string()), float(matrix_entries[5].get_tex_string())],
                [float(matrix_entries[6].get_tex_string()), float(matrix_entries[7].get_tex_string()), float(matrix_entries[8].get_tex_string())]
            ])
            return np.linalg.det(matrix)
        def get_inverse(matrix_entries):
            matrix = np.array([
                [float(matrix_entries[0].get_tex_string()), float(matrix_entries[1].get_tex_string()), float(matrix_entries[2].get_tex_string())],
                [float(matrix_entries[3].get_tex_string()), float(matrix_entries[4].get_tex_string()), float(matrix_entries[5].get_tex_string())],
                [float(matrix_entries[6].get_tex_string()), float(matrix_entries[7].get_tex_string()), float(matrix_entries[8].get_tex_string())]
            ])
            return np.int32(np.linalg.det(matrix)*np.linalg.inv(matrix))
        # Create title
        title = Text("Matrix Inverse Calculation")
        title.to_edge(UP)
        self.play(Write(title))

        matrix_name = MathTex('A =').to_edge(UL).shift(DOWN*2)
        matrix = Matrix(
            [[1, 2, 3],
             [-2, 3, 0],
            [1, 0, 2] ]
        )
        matrix.scale(0.75).next_to(matrix_name, RIGHT)
        self.play(Write(matrix_name), Write(matrix))
        self.wait(1)
        matrix_inverse_name = MathTex(r'A^{-1} = \frac{1}{|A|}').next_to(matrix).shift(RIGHT*2)
        

        matrix_entries = matrix.get_entries()
        inverse_matrix = Matrix(get_inverse(matrix_entries)).scale(0.75).next_to(matrix_inverse_name)
        
        self.play(Write(matrix_inverse_name), *[FadeIn(mobj) for mobj in inverse_matrix.get_brackets()])
        self.wait(1)
        final_det = round(get_determinant_value3(matrix_entries))
        matrix_inverse_name_new = MathTex(fr'A^{{-1}} = \frac{1}{{{final_det}}}').next_to(matrix).shift(RIGHT*2)
        self.play(ReplacementTransform(matrix_inverse_name, matrix_inverse_name_new))
        run_time = 2
        num_rows = 3
        adjoint_dict = {0:[4,5,7,8], 1:[3,5,6,8], 2:[3,4,6,7], 
                        3:[1,2,7,8], 4:[0,2,6,8], 5:[0,1,6,7], 
                        6:[1,2,4,5], 7:[0,2,3,5], 8:[0,1,3,4]}
        for i in range(3):
            for j in range(3):
        
                entry = i*num_rows + j
                inverse_matrix_entries = inverse_matrix.get_entries()
                rect = SurroundingRectangle(inverse_matrix_entries[entry], color=GREEN)
                adj = adjoint_dict[entry]
                m1_values = [
                    [matrix_entries[adj[0]].get_tex_string(), matrix_entries[adj[1]].get_tex_string()],
                    [matrix_entries[adj[2]].get_tex_string(), matrix_entries[adj[3]].get_tex_string()]
                ]

                m1 = Matrix(m1_values,
                    left_bracket="|",
                    right_bracket="|").scale(0.75)
                m1.next_to(matrix, 2.5*DOWN) 
                m1_entries = m1.get_entries()
                self.play(matrix_entries[entry].animate.set_color(GREEN), Create(rect), run_time=run_time)
                
                _row = VGroup(matrix.get_rows()[i])
                _column = VGroup(matrix.get_columns()[j])
                rect1 = SurroundingRectangle(_row, color=BLUE)
                rect2 = SurroundingRectangle(_column, color=BLUE)
                self.play(Create(rect1), Create(rect2), run_time=run_time)
                self.play(matrix_entries[adj[0]].animate.set_color(YELLOW),
                        matrix_entries[adj[1]].animate.set_color(YELLOW),
                        matrix_entries[adj[2]].animate.set_color(YELLOW),
                        matrix_entries[adj[3]].animate.set_color(YELLOW), run_time=run_time)

                self.play(ReplacementTransform(matrix_entries[adj[0]].copy(), m1.get_entries()[0]),
                        ReplacementTransform(matrix_entries[adj[1]].copy(), m1.get_entries()[1]),
                        ReplacementTransform(matrix_entries[adj[2]].copy(), m1.get_entries()[2]),
                        ReplacementTransform(matrix_entries[adj[3]].copy(), m1.get_entries()[3]),
                        *[FadeIn(mobj) for mobj in m1.get_brackets()],run_time = run_time)
                
                det_value = round(get_determinant_value(m1_entries))  # round to get an integer value, adjust as needed
                equal = MathTex(fr"=").next_to(m1, RIGHT)
                det = MathTex(fr"{det_value}").next_to(equal, RIGHT)
                self.play(Write(equal), Write(det), run_time=run_time/2)
                self.play(ReplacementTransform(det, inverse_matrix.get_entries()[entry] ), FadeOut(equal), run_time=run_time)

                self.play(
                    matrix_entries[adj[0]].animate.set_color(WHITE),
                    matrix_entries[adj[1]].animate.set_color(WHITE),
                    matrix_entries[adj[2]].animate.set_color(WHITE),
                    matrix_entries[adj[3]].animate.set_color(WHITE), FadeOut(rect),  FadeOut(m1), FadeOut(rect1), FadeOut(rect2))
                run_time = run_time/2

        self.wait(3)

class cosine_similarity(Scene):

    def construct(self):
        def cosine_similarity_value(v1, v2):
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            return dot_product / (norm_v1 * norm_v2)

        title = Text("Cosine Similarity").to_edge(UP)
        cossim_label = MathTex(r"sim_C(x,y) = \frac{x^Ty}{||x||||y||}").scale(0.75).to_edge(UL).shift(DOWN*2)

        plane = NumberPlane(
            x_range=[-3, 3], 
            y_range=[-3, 3], 
            axis_config={"color": WHITE},
            background_line_style={
                "stroke_color": WHITE,
                "stroke_width": 2,
                "stroke_opacity": 0.1
            }
        )          
        self.play(Write(title))
        self.play(Write(cossim_label))
        self.wait(1)
        x = np.array([1, 2, 0])
        dot = Dot(point=x)
        dot_label = MathTex(r"x").next_to(dot, UP)

        # Calculate normalized x
        x_norm = np.linalg.norm(x[:2])  # Taking first two elements since we're in 2D
        x_hat = x / x_norm

        # Move dot to normalized position
        dot_target = Dot(point=x_hat)
        dot_label_target = MathTex(r"\frac{x}{||x||}").next_to(dot_target, UP)

        y = np.array([-2, 1, 0])
        dot_y = Dot(point=y)
        dot_label_y = MathTex(r"y").next_to(dot_y, UP)

        # Calculate normalized x
        y_norm = np.linalg.norm(y[:2])  # Taking first two elements since we're in 2D
        y_hat = y / y_norm

        # Move dot to normalized position
        dot_target_y = Dot(point=y_hat)
        dot_label_target_y = MathTex(r"\frac{y}{||y||}").next_to(dot_target_y, UP)


        dot_cp = dot.copy()
        dot_y_cp = dot_y.copy()
        dot_label_cp = dot_label.copy()
        dot_label_y_cp = dot_label_y.copy()

        self.play(Create(plane))
        self.play(Create(dot_cp), Write(dot_label_cp))
        self.play(Create(dot_y_cp), Write(dot_label_y_cp))
        self.wait(2)
        self.play(ReplacementTransform(dot_cp, dot_target), ReplacementTransform(dot_label_cp, dot_label_target))
        self.play(ReplacementTransform(dot_y_cp, dot_target_y), ReplacementTransform(dot_label_y_cp, dot_label_target_y))
        self.wait(1)
        circle = Circle(radius=1, color=YELLOW)
        self.play(Create(circle))
        self.wait(2)

        x_vector = Vector(x_hat)
        y_vector = Vector(y_hat)
        self.play(Create(x_vector), Create(y_vector))
        self.wait(2)
        angle_between_x = angle_between_vectors(x_hat, y_hat)
        angle_x = Sector(start_angle=np.arctan(x_hat[1]/x_hat[0]), angle=angle_between_x, radius=0.35, color="LIGHTBLUE")

        self.play(Create(angle_x))
        self.wait(1)
        self.play(FadeOut(x_vector), FadeOut(y_vector), FadeOut(angle_x))
        self.wait(1)

        self.play(FadeOut(circle), FadeOut(dot_label_target_y),FadeOut(dot_label_target))
        self.play(ReplacementTransform(dot_target, dot), ReplacementTransform(dot_target_y, dot_y))
        dot_label_cp = dot_label.copy()
        dot_label_y_cp = dot_label_y.copy()
        self.play(Write(dot_label_cp), Write(dot_label_y_cp))
        self.wait(1)
        sim_value = cosine_similarity_value(x[:2], y[:2])
        cossim_label2 = MathTex(r"sim_C(x,y) =", f" {sim_value:.2f}").scale(0.75).to_edge(UL).shift(DOWN*2)
        self.play(TransformMatchingTex(cossim_label, cossim_label2))
        cossim_label = cossim_label2
        dot_old = dot_y
        dot_label_old = dot_label_y_cp
        for j in np.linspace(-2, 3, 40):
            y = np.array([j, 1, 0])
            dot_y = Dot(point=y)
            dot_label_y = MathTex(r"y").next_to(dot_y, UP)
            sim_value = cosine_similarity_value(x[:2], y[:2])
            cossim_label2 = MathTex(r"sim_C(x,y) =", f" {sim_value:.2f}").scale(0.75).to_edge(UL).shift(DOWN*2)
            self.play(TransformMatchingTex(cossim_label, cossim_label2), ReplacementTransform(dot_old, dot_y),
                       ReplacementTransform(dot_label_old, dot_label_y), run_time=0.1)
            cossim_label = cossim_label2
            dot_old = dot_y
            dot_label_old = dot_label_y
            
        self.wait(1)
        self.play(FadeOut(dot_label_y), FadeOut(cossim_label2), FadeOut(dot_y), FadeOut(dot), FadeOut(dot_label_cp))
        title2 = Text("FaceID with Cosine Similarity").to_edge(UP)
        self.play(ReplacementTransform(title, title2))

        for i in range(1,5):
            image = ImageMobject(f"faces/x{i}.jpeg").to_edge(UR).shift(DOWN)
            self.wait(1)
            self.play(FadeIn(image))
            if i==1:
                cnn_text = Text("CNN", font="Arial", color=WHITE).scale(0.5)
                fancy_box = Rectangle(width=cnn_text.get_width() + 0.5,
                                      height=cnn_text.get_height() + 0.7,
                                      color=WHITE,
                                      fill_color=RED,
                                      fill_opacity=0.8)
                cnn_box = VGroup(fancy_box, cnn_text)
                cnn_box.next_to(image, DOWN, buff=0.5)
                self.wait(1)
                self.play(FadeIn(cnn_box))
            self.play(image.animate.next_to(cnn_box, UP, buff=0),
                      rate_func=linear,
                      run_time=0.7)

            x_text = MathTex(r"x")
            x_text.next_to(cnn_box, DOWN, buff=0.1)  # Place it just below the cnn_box
            self.play(
                FadeOut(image),
                Write(x_text),
                x_text.animate.shift(DOWN*0.5)  # Move the "x" text south by 0.5 units
            )

            x = 2 + random.uniform(-0.5, 0.5)
            y = 1 + random.uniform(-0.5, 0.5)

            target_dot = Dot(point=[x,y,0], color=WHITE)
            self.play(Transform(x_text, target_dot))
        self.wait(2)
        y_locs = [(-1,-2), (-1,1), (1,-2)]
        for i in range(1,4):
            imgname = f"faces/y{i}.jpeg" if i<3 else f"faces/y{i}.jpg"
            image = ImageMobject(imgname)
            if i==3:
                image.scale(0.2)
            image.to_edge(UR).shift(DOWN)
            self.wait(1)
            self.play(FadeIn(image))

            self.play(image.animate.next_to(cnn_box, UP, buff=0),
                      rate_func=linear,
                      run_time=0.7)

            x_text = MathTex(r"x")
            x_text.next_to(cnn_box, DOWN, buff=0.1)  # Place it just below the cnn_box
            self.play(
                FadeOut(image),
                Write(x_text),
                x_text.animate.shift(DOWN*0.5)  # Move the "x" text south by 0.5 units
            )

            x = y_locs[i-1][0] + random.uniform(-0.5, 0.5)
            y = y_locs[i-1][1] + random.uniform(-0.5, 0.5)

            target_dot = Dot(point=[x,y,0], color=RED)
            self.play(Transform(x_text, target_dot))
        self.wait(2)
        imgname = f"faces/x_test.jpeg"
        image = ImageMobject(imgname).to_edge(UR).shift(DOWN)
        self.play(FadeIn(image))
        self.play(image.animate.next_to(cnn_box, UP, buff=0),
                      rate_func=linear,
                      run_time=0.7)

        x_text = MathTex(r"x")
        x_text.next_to(cnn_box, DOWN, buff=0.1)  # Place it just below the cnn_box
        self.play(
                FadeOut(image),
                Write(x_text),
                x_text.animate.shift(DOWN*0.5)  # Move the "x" text south by 0.5 units
            )

        x = 2 + random.uniform(-0.5, 0.5)
        y = 1 + random.uniform(-0.5, 0.5)

        target_dot = Dot(point=[x,y,0], color=GREEN)
        self.play(Transform(x_text, target_dot))




        self.wait(2)
