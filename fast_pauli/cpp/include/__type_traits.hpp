#ifndef __FP_TYPE_TRAITS_HPP
#define __FP_TYPE_TRAITS_HPP

#include <complex>
#include <concepts>
#include <execution>
#include <type_traits>

namespace fast_pauli
{

template <typename T> struct is_complex : std::false_type
{
};

template <std::floating_point T> struct is_complex<std::complex<T>> : std::true_type
{
};

/**
 * @brief Execution policy concept
 *
 * @tparam T
 */
template <typename T>
concept execution_policy = std::is_execution_policy_v<std::remove_cvref_t<T>>;

/**
 * @brief Parallel execution policy helpers
 *
 * Modelled after libcxx implementation
 * https://github.com/llvm/llvm-project/blob/6bed79b3f00d3e2c273bc36ed350f802d76607b3/libcxx/include/execution#L96-L118
 *
 * @tparam class
 */

template <typename T>
inline constexpr bool is_parallel_execution_policy_v =
    std::is_same_v<std::execution::parallel_policy, std::remove_cvref_t<T>>;

} // namespace fast_pauli

#endif
