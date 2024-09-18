#ifndef __PAULI_HPP
#define __PAULI_HPP

#include <fmt/format.h>

#include <cinttypes>
#include <complex>
#include <concepts>
#include <span>
#include <utility>
#include <vector>

using namespace std::literals;

namespace fast_pauli {

/**
 * @brief A class for efficient representation of a
 * 2x2 Pauli matrix \f$ \sigma_i \in \{ I,X,Y,Z \} \f$
 *
 */
struct Pauli
{
  uint8_t code; // 0: I, 1: X, 2: Y, 3: Z

  /**
   * @brief Default constructor, initializes to I.
   *
   */
  constexpr Pauli()
    : code(0)
  {
  }

  /**
   * @brief Constructor given a numeric code.
   *
   * @tparam T Any type convertible to uint8_t
   * @param code 0: I, 1: X, 2: Y, 3: Z
   * @return requires
   */
  template<class T>
    requires std::convertible_to<T, uint8_t>
  constexpr Pauli(T const code)
    : code(code)
  {
    if (code < 0 || code > 3)
      throw std::invalid_argument("Pauli code must be 0, 1, 2, or 3");
  }

  /**
   * @brief Constructor given Pauli matrix symbol.
   *
   * @param symbol pauli matrix - I, X, Y, Z
   * @return
   */
  constexpr Pauli(char const symbol)
  {
    switch (symbol) {
      case 'I':
        this->code = 0;
        break;
      case 'X':
        this->code = 1;
        break;
      case 'Y':
        this->code = 2;
        break;
      case 'Z':
        this->code = 3;
        break;
      default:
        throw std::invalid_argument("Invalid Pauli matrix symbol");
    }
  }

  // Copy ctor
  Pauli(Pauli const& other) = default;

  // Copy assignment
  Pauli& operator=(Pauli const& other) noexcept = default;

  // The default operator does everything we want and make this more composable
  friend auto operator<=>(Pauli const&, Pauli const&) = default;

  /**
   * @brief Returns the product of two pauli matrices and their phase as a pair.
   *
   * @param lhs left hand side pauli object
   * @param rhs right hand side pauli object
   * @return  std::pair<std::complex<double>, Pauli> phase and resulting pauli
   * matrix
   */
  friend std::pair<std::complex<double>, Pauli> operator*(Pauli const& lhs, Pauli const& rhs)
  {
    switch (lhs.code) {
      case 0:
        switch (rhs.code) {
          case 0:
            return { 1, Pauli{ 0 } }; // I * I = I
          case 1:
            return { 1, Pauli{ 1 } }; // I * X = X
          case 2:
            return { 1, Pauli{ 2 } }; // I * Y = Y
          case 3:
            return { 1, Pauli{ 3 } }; // I * Z = Z
          default:
            // Should never reach here
            throw std::runtime_error("Unexpected Pauli code");
        }
      case 1:
        switch (rhs.code) {
          case 0:
            return { 1, Pauli{ 1 } }; // X * I = X
          case 1:
            return { 1, Pauli{ 0 } }; // X * X = I
          case 2:
            return { 1i, Pauli{ 3 } }; // X * Y = iZ
          case 3:
            return { -1i, Pauli{ 2 } }; // X * Z = -iY
          default:
            // Should never reach here
            throw std::runtime_error("Unexpected Pauli code");
        }
      case 2:
        switch (rhs.code) {
          case 0:
            return { 1, Pauli{ 2 } }; // Y * I = Y
          case 1:
            return { -1i, Pauli{ 3 } }; // Y * X = -iZ
          case 2:
            return { 1, Pauli{ 0 } }; // Y * Y = I
          case 3:
            return { 1i, Pauli{ 1 } }; // Y * Z = iX
          default:
            // Should never reach here
            throw std::runtime_error("Unexpected Pauli code");
        }
      case 3:
        switch (rhs.code) {
          case 0:
            return { 1, Pauli{ 3 } }; // Z * I = Z
          case 1:
            return { 1i, Pauli{ 2 } }; // Z * X = iY
          case 2:
            return { -1i, Pauli{ 1 } }; // Z * Y = -iX
          case 3:
            return { 1, Pauli{ 0 } }; // Z * Z = I
          default:
            // Should never reach here
            throw std::runtime_error("Unexpected Pauli code");
        }
      default:
        // Should never reach here
        throw std::runtime_error("Unexpected Pauli code");
    }
  }

  /**
   * @brief Returns the pauli matrix as a 2D vector of complex numbers.
   *
   * @tparam T floating point type
   * @return  std::vector<std::vector<std::complex<T>>>
   */
  template<std::floating_point T>
  std::vector<std::vector<std::complex<T>>> to_tensor() const
  {
    std::vector<std::vector<std::complex<T>>> result;
    switch (code) {
      case 0:
        result = { { 1, 0 }, { 0, 1 } };
        break;
      case 1:
        result = { { 0, 1 }, { 1, 0 } };
        break;
      case 2:
        result = { { 0, -1i }, { 1i, 0 } };
        break;
      case 3:
        result = { { 1, 0 }, { 0, -1 } };
        break;
      default:
        throw std::runtime_error("Unexpected Pauli code");
        break;
    }
    return result;
  }
};

} // namespace fast_pauli

// Adding specialization to the fmt library so we can easily print Pauli
template<>
struct fmt::formatter<fast_pauli::Pauli>
{
  constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

  template<typename FormatContext>
  auto format(fast_pauli::Pauli const& p, FormatContext& ctx) const
  {
    char code_char;
    switch (p.code) {
      case 0:
        code_char = 'I';
        break;
      case 1:
        code_char = 'X';
        break;
      case 2:
        code_char = 'Y';
        break;
      case 3:
        code_char = 'Z';
        break;
      default:
        throw std::runtime_error("Unexpected Pauli code");
        break;
    }
    return fmt::format_to(ctx.out(), "{}", code_char);
  }
};

// Add complex numbers because they aren't in there already, TODO __pauli.hpp
// may not be the best place for these
template<>
struct fmt::formatter<std::complex<double>>
{
  constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

  template<typename FormatContext>
  auto format(std::complex<double> const& v, FormatContext& ctx) const
  {
    return fmt::format_to(ctx.out(), "({}, {}i)", v.real(), v.imag());
  }
};

#endif // __PAULI_HPP
