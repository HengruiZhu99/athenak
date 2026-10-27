// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_POSITIVE_FIELD_CHART_HPP_
#define Z4C_HYPERBOLOIDAL_POSITIVE_FIELD_CHART_HPP_

#include <limits>
#include "z4c/hyperboloidal/conformal_rhs.hpp"

namespace z4c { namespace hyperboloidal {

enum class PositiveFieldMode { direct, log_reference, square };
enum class PositiveFieldStatus {
  valid, invalid_mode, invalid_reference, negative_value, nonfinite_input,
  endpoint_data_required, underflow, overflow, subnormal_value
};

KOKKOS_INLINE_FUNCTION
const char *PositiveFieldStatusName(PositiveFieldStatus status) {
  switch(status) {
    case PositiveFieldStatus::valid:return "valid";
    case PositiveFieldStatus::invalid_mode:return "invalid_mode";
    case PositiveFieldStatus::invalid_reference:return "invalid_reference";
    case PositiveFieldStatus::negative_value:return "negative_value";
    case PositiveFieldStatus::nonfinite_input:return "nonfinite_input";
    case PositiveFieldStatus::endpoint_data_required:return "endpoint_data_required";
    case PositiveFieldStatus::underflow:return "underflow";
    case PositiveFieldStatus::overflow:return "overflow";
    case PositiveFieldStatus::subnormal_value:return "subnormal_value";
  }
  return "invalid_mode";
}

template<typename T> KOKKOS_INLINE_FUNCTION
ScalarJet<T> UnitPositiveFieldReference() {ScalarJet<T> ref{};ref.value=T(1);return ref;}

// A representation of a positive physical scalar, with NO assumed radial power.
// This class changes neither a gauge/source nor the spatial differencing rule.
// log_reference uses a known positive reference jet and its fixed-x time rate;
// a moving reference must include that time rate, never omit it.
//
// No floor or clamp is applied. All outputs remain unchanged on error. Direct
// storage can carry an explicitly supplied zero-endpoint jet; inverse log/square
// derivatives at zero require independent limiting data and are rejected. Normal
// representable values are the supported arithmetic contract. Subnormal scaled
// values are explicitly reported rather than silently accepted with lost digits.
template<typename T> struct PositiveFieldChart {
  PositiveFieldMode mode=PositiveFieldMode::direct;
  ScalarJet<T> reference=UnitPositiveFieldReference<T>();
  T reference_time=T(0);

  KOKKOS_INLINE_FUNCTION PositiveFieldChart()=default;
  KOKKOS_INLINE_FUNCTION explicit PositiveFieldChart(PositiveFieldMode m):mode(m) {}
  KOKKOS_INLINE_FUNCTION PositiveFieldChart(PositiveFieldMode m,const ScalarJet<T> &ref,T rt=T(0))
      :mode(m),reference(ref),reference_time(rt) {}

  KOKKOS_INLINE_FUNCTION PositiveFieldStatus Encode(const ScalarJet<T> &physical,
                                                   ScalarJet<T> &stored) const {
    auto status=Validate();if(status!=PositiveFieldStatus::valid)return status;
    if(!Finite(physical))return PositiveFieldStatus::nonfinite_input;
    status=PhysicalInput(physical.value);if(status!=PositiveFieldStatus::valid)return status;
    ScalarJet<T> q{};
    if(mode==PositiveFieldMode::direct){stored=physical;return PositiveFieldStatus::valid;}
    if(mode==PositiveFieldMode::log_reference){
      // Avoid a spurious overflow/underflow in physical/reference before log.
      q.value=Kokkos::log(physical.value)-Kokkos::log(reference.value);
      T a[3],b[3];for(int i=0;i<3;++i){a[i]=physical.d[i]/physical.value;b[i]=reference.d[i]/reference.value;q.d[i]=a[i]-b[i];}
      for(int i=0;i<3;++i)for(int j=0;j<3;++j)
        q.dd[i][j]=physical.dd[i][j]/physical.value-a[i]*a[j]
                    -reference.dd[i][j]/reference.value+b[i]*b[j];
    }else{
      q.value=physical.value*physical.value;
      status=PositiveResult(q.value);if(status!=PositiveFieldStatus::valid)return status;
      for(int i=0;i<3;++i){q.d[i]=T(2)*physical.value*physical.d[i];
        if(q.d[i]==T(0)&&physical.d[i]!=T(0))return PositiveFieldStatus::underflow;
        for(int j=0;j<3;++j)q.dd[i][j]=T(2)*(physical.d[i]*physical.d[j]+physical.value*physical.dd[i][j]);}
    }
    if(!Finite(q))return PositiveFieldStatus::overflow;
    stored=q;return PositiveFieldStatus::valid;
  }

  KOKKOS_INLINE_FUNCTION PositiveFieldStatus Reconstruct(const ScalarJet<T> &stored,
                                                        ScalarJet<T> &physical) const {
    auto status=Validate();if(status!=PositiveFieldStatus::valid)return status;
    // A log value -infinity denotes zero but does not supply a derivative limit.
    if(mode==PositiveFieldMode::log_reference&&stored.value==-std::numeric_limits<T>::infinity())
      return PositiveFieldStatus::endpoint_data_required;
    if(!Finite(stored))return PositiveFieldStatus::nonfinite_input;
    ScalarJet<T> q{};status=PhysicalValue(stored.value,q.value);
    if(status!=PositiveFieldStatus::valid)return status;
    if(mode==PositiveFieldMode::direct){physical=stored;return PositiveFieldStatus::valid;}
    if(mode==PositiveFieldMode::log_reference){
      T a[3],b[3];for(int i=0;i<3;++i){b[i]=reference.d[i]/reference.value;a[i]=stored.d[i]+b[i];q.d[i]=q.value*a[i];
        if(q.d[i]==T(0)&&a[i]!=T(0))return PositiveFieldStatus::underflow;}
      for(int i=0;i<3;++i)for(int j=0;j<3;++j){
        const T h=stored.dd[i][j]+reference.dd[i][j]/reference.value-b[i]*b[j]+a[i]*a[j];
        q.dd[i][j]=q.value*h;
        if(q.dd[i][j]==T(0)&&h!=T(0))return PositiveFieldStatus::underflow;
      }
    }else{
      for(int i=0;i<3;++i)q.d[i]=stored.d[i]/(T(2)*q.value);
      // Avoid explicitly forming F^3, which could underflow although F^2 is
      // representable. Cancellation remains a conditioning issue, not a reset.
      for(int i=0;i<3;++i)for(int j=0;j<3;++j)
        q.dd[i][j]=stored.dd[i][j]/(T(2)*q.value)-q.d[i]*q.d[j]/q.value;
    }
    if(!Finite(q))return PositiveFieldStatus::overflow;
    physical=q;return PositiveFieldStatus::valid;
  }

  KOKKOS_INLINE_FUNCTION PositiveFieldStatus EncodeRate(T physical_value,T physical_rate,
                                                        T &stored_rate) const {
    auto status=Validate();if(status!=PositiveFieldStatus::valid)return status;
    if(!Kokkos::isfinite(physical_rate))return PositiveFieldStatus::nonfinite_input;
    status=PhysicalInput(physical_value);if(status!=PositiveFieldStatus::valid)return status;
    T rate=physical_rate;
    if(mode==PositiveFieldMode::log_reference)rate=physical_rate/physical_value-reference_time/reference.value;
    else if(mode==PositiveFieldMode::square){
      const T scaled=physical_value*physical_value;
      status=PositiveResult(scaled);if(status!=PositiveFieldStatus::valid)return status;
      rate=T(2)*physical_value*physical_rate;
      if(rate==T(0)&&physical_rate!=T(0))return PositiveFieldStatus::underflow;
    }
    if(!Kokkos::isfinite(rate))return PositiveFieldStatus::overflow;
    stored_rate=rate;return PositiveFieldStatus::valid;
  }

  KOKKOS_INLINE_FUNCTION PositiveFieldStatus ReconstructRate(T stored_value,T stored_rate,
                                                             T &physical_rate) const {
    auto status=Validate();if(status!=PositiveFieldStatus::valid)return status;
    if(!Kokkos::isfinite(stored_rate))return PositiveFieldStatus::nonfinite_input;
    T value;status=PhysicalValue(stored_value,value);if(status!=PositiveFieldStatus::valid)return status;
    T rate=stored_rate;
    if(mode==PositiveFieldMode::log_reference){
      const T relative=stored_rate+reference_time/reference.value;rate=value*relative;
      if(rate==T(0)&&relative!=T(0))return PositiveFieldStatus::underflow;
    }else if(mode==PositiveFieldMode::square)rate=stored_rate/(T(2)*value);
    if(!Kokkos::isfinite(rate))return PositiveFieldStatus::overflow;
    physical_rate=rate;return PositiveFieldStatus::valid;
  }

  // Absolute reconstruction Jacobian, useful alongside the physical residual.
  // It is a diagnostic, not a proof of well-conditioned spatial differencing.
  KOKKOS_INLINE_FUNCTION PositiveFieldStatus ReconstructionJacobian(T stored_value,T &jacobian) const {
    auto status=Validate();if(status!=PositiveFieldStatus::valid)return status;
    T value;status=PhysicalValue(stored_value,value);if(status!=PositiveFieldStatus::valid)return status;
    const T q=mode==PositiveFieldMode::direct?T(1):(mode==PositiveFieldMode::log_reference?value:T(1)/(T(2)*value));
    if(!Kokkos::isfinite(q))return PositiveFieldStatus::overflow;
    jacobian=q;return PositiveFieldStatus::valid;
  }

 private:
  KOKKOS_INLINE_FUNCTION static bool Finite(const ScalarJet<T> &q) {
    bool valid=Kokkos::isfinite(q.value);
    for(int i=0;i<3;++i){valid=valid&&Kokkos::isfinite(q.d[i]);for(int j=0;j<3;++j)valid=valid&&Kokkos::isfinite(q.dd[i][j]);}
    return valid;
  }
  KOKKOS_INLINE_FUNCTION PositiveFieldStatus Validate() const {
    if(mode!=PositiveFieldMode::direct&&mode!=PositiveFieldMode::log_reference&&mode!=PositiveFieldMode::square)
      return PositiveFieldStatus::invalid_mode;
    if(mode==PositiveFieldMode::log_reference&&(!(reference.value>T(0))||!Finite(reference)||!Kokkos::isfinite(reference_time)))
      return PositiveFieldStatus::invalid_reference;
    return PositiveFieldStatus::valid;
  }
  KOKKOS_INLINE_FUNCTION PositiveFieldStatus PhysicalInput(T value) const {
    if(!Kokkos::isfinite(value))return PositiveFieldStatus::nonfinite_input;
    if(value<T(0))return PositiveFieldStatus::negative_value;
    if(value==T(0))return mode==PositiveFieldMode::direct?PositiveFieldStatus::valid:PositiveFieldStatus::endpoint_data_required;
    if(value<std::numeric_limits<T>::min())return PositiveFieldStatus::subnormal_value;
    return PositiveFieldStatus::valid;
  }
  KOKKOS_INLINE_FUNCTION static PositiveFieldStatus PositiveResult(T value) {
    if(!Kokkos::isfinite(value))return PositiveFieldStatus::overflow;
    if(value==T(0))return PositiveFieldStatus::underflow;
    if(value<T(0))return PositiveFieldStatus::negative_value;
    if(value<std::numeric_limits<T>::min())return PositiveFieldStatus::subnormal_value;
    return PositiveFieldStatus::valid;
  }
  KOKKOS_INLINE_FUNCTION PositiveFieldStatus PhysicalValue(T stored_value,T &value) const {
    if(mode==PositiveFieldMode::log_reference&&stored_value==-std::numeric_limits<T>::infinity())
      return PositiveFieldStatus::endpoint_data_required;
    if(!Kokkos::isfinite(stored_value))return PositiveFieldStatus::nonfinite_input;
    T q;
    if(mode==PositiveFieldMode::direct){auto status=PhysicalInput(stored_value);if(status!=PositiveFieldStatus::valid)return status;q=stored_value;}
    else if(mode==PositiveFieldMode::log_reference){q=Kokkos::exp(Kokkos::log(reference.value)+stored_value);
      auto status=PositiveResult(q);if(status!=PositiveFieldStatus::valid)return status;}
    else{if(stored_value==T(0))return PositiveFieldStatus::endpoint_data_required;
      auto status=PositiveResult(stored_value);if(status!=PositiveFieldStatus::valid)return status;q=Kokkos::sqrt(stored_value);}
    value=q;return PositiveFieldStatus::valid;
  }
};

}}  // namespace z4c::hyperboloidal
#endif
