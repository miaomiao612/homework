'''
Author: miaomiao612 dddoctorr612@gmail.com
Date: 2023-02-07 02:51:45
LastEditors: miaomiao612 dddoctorr612@gmail.com
LastEditTime: 2023-02-08 00:25:52
FilePath: \week3\test_model.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import inspect
from model import MLP


def test_has_correct_attributes():
    """
    Test that the LinearRegression class has the correct attributes.
    """
    lr = MLP(1, 1, 1)
    assert hasattr(lr, "forward"), f"{str(MLP)} does not have method `forward`."


def test_fn_signatures():
    """
    Disallow untyped signatures.
    """
    from inspect import signature

    lr = MLP(1, 1, 1)
    # all methods' arguments and returns must be typed.
    methods = ["forward"]
    for method in methods:
        assert (
            signature(getattr(lr, method)).return_annotation is not inspect._empty
        ), f"The return type of `{method}` is not annotated."

        # Arguments must be typed.
        for param in signature(getattr(lr, method)).parameters.values():
            assert (
                param.annotation is not inspect._empty
            ), f"The argument type of `{method}:{param.name}` is not annotated."


def test_docstrings():
    """
    Disallow missing docstrings.
    """
    lr = MLP(1, 1, 1)
    # all methods must have a docstring.
    methods = ["forward"]
    for method in methods:
        assert (
            getattr(lr, method).__doc__ is not None
        ), f"The method `{method}` does not have a docstring."

    # all classes must have a docstring.
    classes = [MLP]
    for class_ in classes:
        assert (
            class_.__doc__ is not None
        ), f"The class `{class_}` does not have a docstring."