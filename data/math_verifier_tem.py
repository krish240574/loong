# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========

from typing import Optional
import logging
import re

from camel.extractors.base import BaseExtractor
from camel.logger import get_logger
from camel.verifiers import BaseVerifier
from camel.verifiers.models import VerificationOutcome, VerificationResult
from math_verify import parse, verify
from math_verify.parser import LatexExtractionConfig
from sympy import sympify

logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)

class MathVerifier(BaseVerifier):
    r"""Verifier for mathematical expressions using Math-Verify.

    Features:
    - Supports LaTeX and plain mathematical expressions
    - Handles complex numbers, matrices, and sets
    - Configurable precision for floating-point comparisons
    - Optional LaTeX wrapping to ensure proper parsing and rendering
    - Comprehensive error handling and logging
    """

    def __init__(
        self,
        extractor: Optional[BaseExtractor] = None,
        timeout: Optional[float] = 30.0,
        float_rounding: int = 6,
        numeric_precision: int = 15,
        **kwargs,
    ):
        r"""Initializes the MathVerifier.

        Args:
            extractor (Optional[BaseExtractor], optional): The extractor to use
                for extracting code from the solution. (default: :obj:`None`)
            timeout (Optional[float], optional): The execution timeout in
                seconds. (default: :obj:`30.0`)
            float_rounding (Optional[int], optional): The number of decimal 
                places to round floating-point numbers. (default: :obj:`6`)
            numeric_precision (Optional[int], optional): The numeric precision 
                for floating-point comparisons. (default: :obj:`15`)
        """
        super().__init__(extractor=extractor, timeout=timeout, **kwargs)
        self.float_rounding = float_rounding
        self.numeric_precision = numeric_precision

    async def _setup(self, **kwargs) -> None:
        r"""No special setup needed for math verification."""
        pass

    async def _cleanup(self) -> None:
        r"""No cleanup needed for math verification."""
        pass

    @staticmethod
    def _latex_wrapping(s: str) -> str:
        r"""Wrap a LaTeX expression in math mode delimiters.

        This function checks whether the input string is already in a LaTeX
        math environment (e.g., $, \[, \begin{}, etc.). If not, it wraps the
        expression in $$...$$ to ensure proper parsing and rendering as a
        mathematical expression.

        Args:
            s (str): The input LaTeX string.

        Returns:
            str: The LaTeX string wrapped in math mode if necessary.
        """
        s_stripped = s.strip()
        if not (re.search(r'\$\s*.*?\s*\$', s_stripped) or 
                s_stripped.startswith("\\(") or 
                s_stripped.startswith("\\[") or 
                s_stripped.startswith("\\begin")):
            s = f"$$ {s_stripped} $$"
        return s

    @staticmethod
    def _any_equivalent_pair(
        ref_list,
        sol_list,
        float_rounding=6,
        numeric_precision=15,
    ) -> bool:
        r"""Check if any pair of expressions from two lists are equivalent.

        This function iterates over every combination of expressions from the 
        reference and solution lists. It uses a primary verification routine 
        to check for equivalence, and if that fails, it attempts a numerical 
        comparison by converting the expressions into their evaluated forms.
        As a fallback, it compares the string representations after stripping 
        to determine if any expression appears identical.

        Args:
            ref_list (list): A list of reference expressions.
            sol_list (list): A list of solution expressions.
            float_rounding (Optional[int], optional): The number of decimal
                places to round floating-point numbers. (default: :obj:`6`)
            numeric_precision (Optional[int], optional): The precision for
                numerical evaluations. (default: :obj:`15`)

        Returns:
            bool: True if at least one equivalent expression pair is found;
                otherwise, False. """
        for ref in ref_list:
            for sol in sol_list:
                try:
                    if verify(
                        ref,
                        sol,
                        float_rounding=float_rounding,
                        numeric_precision=numeric_precision,
                    ):
                        return True
                except Exception:
                    pass

                try:
                    ref_numeric = sympify(ref).evalf()
                    sol_numeric = sympify(sol).evalf()
                    tolerance = 10 ** (-float_rounding)
                    if abs(ref_numeric - sol_numeric) < tolerance:
                        return True
                except Exception:
                    pass

        ref_strs = {str(r).strip() for r in ref_list}
        sol_strs = {str(s).strip() for s in sol_list}
        return bool(ref_strs & sol_strs)


    @staticmethod
    def _latex_to_expr_equiv(s: str) -> str:
        r"""Convert LaTeX math expressions to equivalent Python expressions.

        This function transforms LaTeX mathematical notation into a format that can 
        be parsed by symbolic mathematics libraries like SymPy. It handles common
        LaTeX commands such as \sqrt, \frac, \cdot, and trigonometric functions.

        Args:
            s (str): The LaTeX mathematical expression to convert.

        Returns:
            str: The transformed expression in Python-compatible format.
        """
        sqrt_pattern = re.compile(r'\\sqrt\{([^{}]+?)\}')
        frac_pattern = re.compile(
            r'\\(?:dfrac|tfrac|frac)\s*\{([^{}]+?)\}\s*\{([^{}]+?)\}'
        )
        
        replacement_patterns = [
            (re.compile(r'\\cdot'), '*'),
            (re.compile(r'\\times'), '*'),
            (re.compile(r'\\pi'), 'pi'),
            (re.compile(r'\\mathrm\{e\}'), 'e'),
            (re.compile(r'\\sin'), 'sin'),
            (re.compile(r'\\cos'), 'cos'),
            (re.compile(r'\\tan'), 'tan'),
            (re.compile(r'\\arcsin'), 'asin'),
            (re.compile(r'\\arccos'), 'acos'),
            (re.compile(r'\\arctan'), 'atan'),
        ]

        power_pattern = re.compile(r'(?<=[0-9a-zA-Z\)])\^')
        
        s = sqrt_pattern.sub(r'sqrt(\1)', s)
        s = frac_pattern.sub(r'(\1)/(\2)', s)
        for pattern, replacement in replacement_patterns:
            s = pattern.sub(replacement, s)
        s = power_pattern.sub('**', s)
        
        return s

    def check_equivalence(self, ref_list, sol_list):
        r"""Check if expressions from two lists are mathematically equivalent.

        This function determines if any expression from the reference list is
        equivalent to any expression from the solution list. It performs both
        symbolic verification using the math_verify library and numerical
        comparison when necessary.

        Args:
            ref_list: List of reference expressions to compare against
            sol_list: List of solution expressions to verify

        Returns:
            bool: True if any pair are equivalent, False otherwise
        """
        if ref_list and sol_list:
            if self._any_equivalent_pair(
                ref_list,
                sol_list,
                float_rounding=self.float_rounding,
                numeric_precision=self.numeric_precision,
            ):
                return True
            if verify(
                ref_list,
                sol_list,
                float_rounding=self.float_rounding,
                numeric_precision=self.numeric_precision,
            ):
                return True
        return False

    async def _verify_implementation(
        self, solution: str, reference_answer: Optional[str]
    ) -> VerificationResult:
        r"""Verify mathematical expressions using Math-Verify.

        Args:
            solution: The solution to verify
            reference_answer: The expected answer to compare against

        Returns:
            VerificationResult containing the verification status and details
        """
        if reference_answer is None:
            return VerificationResult(
                status=VerificationOutcome.ERROR,
                result="",
                error_message="Ground truth is required for verification",
            )

        try:
            sol_wrapped = self._latex_wrapping(solution)
            ref_wrapped = self._latex_wrapping(reference_answer)

            parsed_sol_1 = parse(
                sol_wrapped,
                extraction_config=[LatexExtractionConfig()],
            )
            parsed_ref_1 = parse(
                ref_wrapped,
                extraction_config=[LatexExtractionConfig()],
            )

            if self.check_equivalence(parsed_ref_1, parsed_sol_1):
                return VerificationResult(
                    status=VerificationOutcome.SUCCESS, result=solution
                )

            sol_expr = self._latex_to_expr_equiv(solution)
            ref_expr = self._latex_to_expr_equiv(reference_answer)

            try:
                sol_expr_sympy = sympify(sol_expr)
                ref_expr_sympy = sympify(ref_expr)
            except Exception as e:
                logger.debug(f"sympify conversion error: {e}")
                sol_expr_sympy = sol_expr
                ref_expr_sympy = ref_expr

            parsed_sol_2 = [sol_expr_sympy]
            parsed_ref_2 = [ref_expr_sympy]

            if self.check_equivalence(parsed_ref_2, parsed_sol_2):
                return VerificationResult(
                    status=VerificationOutcome.SUCCESS, result=solution
                )

            return VerificationResult(
                status=VerificationOutcome.FAILURE,
                result=solution,
                error_message="Solution does not match ground truth",
            )

        except Exception as error:
            logger.error(f"Mathematical verification error: {error!s}")
            return VerificationResult(
                status=VerificationOutcome.ERROR,
                result="",
                error_message=f"Mathematical verification error: {error!s}",
            )
