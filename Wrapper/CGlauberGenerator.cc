/*
 * COLA Wrapper for CERN ROOT's TGlauber implementation of Glauber model
 * Copyright (C) 2024-2025 Savva Savenkov
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "CGlauberGenerator.hh"

#include "GlauberConst.hh"

#include <cmath>
#include <iterator>
#include <map>
#include <stdexcept>
#include <unordered_map>

namespace {

  cola::AZ DefinedIons(const std::string& name) {
    static const std::unordered_map<std::string, cola::AZ> k_ion_map = {
        {"p", {1, 1}},         {"pg", {1, 1}},       {"pdg", {1, 1}},       {"dpf", {2, 1}},      {"dh", {2, 1}},
        {"d", {2, 1}},         {"He3", {3, 1}},      {"H3", {3, 2}},        {"He4", {4, 2}},      {"C", {12, 6}},
        {"O", {16, 8}},        {"O2", {16, 8}},      {"Oth", {16, 8}},      {"Opar", {16, 8}},    {"Oho", {16, 8}},
        {"Ne", {20, 10}},      {"Ne22", {22, 10}},   {"Necl", {20, 10}},    {"Ne22cl", {22, 10}}, {"Nepgcm", {20, 10}},
        {"Nenleft", {20, 10}}, {"Al", {27, 13}},     {"Si", {28, 14}},      {"Si2", {28, 14}},    {"S", {32, 16}},
        {"Ar", {40, 18}},      {"Ca", {40, 20}},     {"Ca2", {40, 20}},     {"Ni", {58, 28}},     {"Cu", {63, 29}},
        {"Curw ", {63, 29}},   {"Cu2", {63, 29}},    {"Cu2rw", {63, 29}},   {"CuHN", {63, 29}},   {"Br", {79, 35}},
        {"Ag", {109, 47}},     {"Xe", {129, 54}},    {"Xes", {129, 54}},    {"Xe2", {129, 54}},   {"Xe2a", {129, 54}},
        {"Xerw", {129, 54}},   {"Xesrw", {129, 54}}, {"Xe2arw", {129, 54}}, {"W", {186, 74}},     {"Au", {197, 79}},
        {"Aurw", {197, 79}},   {"Au2", {197, 79}},   {"Au2rw", {197, 79}},  {"AuHN", {197, 79}},  {"Pb", {208, 82}},
        {"Pbrw", {208, 82}},   {"Pb*", {208, 82}},   {"PbHN", {208, 82}},   {"Pbpn", {208, 82}},  {"Pbpnrw", {208, 82}},
        {"U", {238, 92}},      {"U2", {238, 92}},
    };

    const auto it = k_ion_map.find(name);
    if (it == k_ion_map.end()) {
      throw std::runtime_error("No such ion in CGlauber");
    }
    return it->second;
  }

  cola::ParticleClass ParticleClassFromNucleon(const TGlauNucleon* nucleon) {
    if (nucleon->IsInNucleusA()) {
      return nucleon->IsSpectator() ? cola::ParticleClass::kSpectatorA : cola::ParticleClass::kNonelasticA;
    }
    return nucleon->IsSpectator() ? cola::ParticleClass::kSpectatorB : cola::ParticleClass::kNonelasticB;
  }

}  // namespace

std::unique_ptr<cola::EventData> CGlauberGenerator::operator()() {
  generator_->Run(1);
  const auto* ev = generator_->GetEvent();
  auto* t_nucleons = generator_->GetNucleons();
  int nev = 0;
  if (t_nucleons != nullptr) {
    nev = t_nucleons->GetEntries();
  }

  auto c_particles = cola::EventParticles(nev);
  auto f_momentum_a = fermi_momentum_->GetMomentum(generator_->GetNucleusA()->GetN(), generator_->GetNpartA());
  auto f_momentum_b = fermi_momentum_->GetMomentum(generator_->GetNucleusB()->GetN(), generator_->GetNpartB());

  const double rapidity = std::log(energy_ + p_za_) - std::log(gconst::kNucleonAverMass);

  f_momentum_a.BoostAxisRapidity(rapidity);
  if (p_zb_ != 0) {
    f_momentum_b.BoostAxisRapidity(-rapidity);
  }

  for (int i = 0; i < nev; ++i) {
    auto* t_nucleon = dynamic_cast<TGlauNucleon*>(t_nucleons->At(i));
    c_particles[i] = cola::Particle{
        .position = cola::LorentzVector{.t = 0, .x = t_nucleon->GetX(), .y = t_nucleon->GetY(), .z = t_nucleon->GetZ()},
        .momentum = t_nucleon->IsInNucleusA() ? f_momentum_a : f_momentum_b,
        .pdg_code = t_nucleon->GetType() != 0 ? 2212 : 2112,
        .p_class = ParticleClassFromNucleon(t_nucleon),
    };
  }
  return std::make_unique<cola::EventData>(cola::EventData{
      .ini_state =
          cola::EventIniState{
              .pdg_code_a = pdg_code_a_,
              .pdg_code_b = pdg_code_b_,
              .pz_a = p_za_,
              .pz_b = p_zb_,
              .energy = is_collider_ ? s_nn_ : energy_,
              .sect_nn = static_cast<float>(x_sect_nn_),
              .b = static_cast<float>(generator_->GetB()),
              .num_coll = generator_->GetNcoll(),
              .num_coll_pp = generator_->GetNcollpp(),
              .num_coll_pn = generator_->GetNcollpn(),
              .num_coll_nn = generator_->GetNcollnn(),
              .num_part = generator_->GetNpart(),
              .num_part_a = generator_->GetNpartA(),
              .num_part_b = generator_->GetNpartB(),
              .phi_rot_a = ev->PhiA,
              .theta_rot_a = ev->ThetaA,
              .phi_rot_b = ev->PhiB,
              .theta_rot_b = ev->ThetaB,
              .ini_state_particles = c_particles,
          },
      .particles = c_particles,
  });
}

CGlauberGenerator::CGlauberGenerator(const std::string& name_a, const std::string& name_b, double energy,
                                     const bool is_collider, std::unique_ptr<FermiMomentum>&& fermi_momentum)
    : is_collider_(is_collider), fermi_momentum_(std::move(fermi_momentum)) {
  energy *= gconst::kGeV;

  const cola::AZ aza = DefinedIons(name_a);
  const cola::AZ azb = DefinedIons(name_b);
  pdg_code_a_ = cola::AZToPdg(aza);
  pdg_code_b_ = cola::AZToPdg(azb);

  const double nucleon_mass = gconst::kNucleonAverMass;
  const double nucleon_mass_sq = nucleon_mass * nucleon_mass;

  if (is_collider_) {
    p_za_ = std::sqrt(energy * energy * 0.25 - nucleon_mass_sq);
    p_zb_ = -p_za_;
    energy_ = energy / 2.0 - nucleon_mass;
    s_nn_ = energy;
  } else {
    p_za_ = std::sqrt(energy * (energy + 2 * nucleon_mass) + nucleon_mass_sq);
    p_zb_ = 0;
    energy_ = energy;
    s_nn_ = std::sqrt(2 * nucleon_mass_sq + 2 * energy * nucleon_mass);
  }

  const double en_at_rest = is_collider_ ? s_nn_ * s_nn_ / 2 / nucleon_mass - 2 * nucleon_mass : energy;
  if (en_at_rest < 425 * gconst::kGeV) {
    const std::map<double, double> bystricky = {
        {0.28, 0.027},  {0.29, 0.051},  {0.30, 0.082},  {0.31, 0.120},  {0.32, 0.171},  {0.34, 0.321},  {0.36, 0.563},
        {0.38, 0.921},  {0.40, 1.41},   {0.42, 2.05},   {0.44, 2.83},   {0.46, 3.74},   {0.48, 4.76},   {0.50, 5.87},
        {0.52, 7.04},   {0.56, 9.48},   {0.60, 11.89},  {0.64, 14.14},  {0.68, 16.15},  {0.72, 17.89},  {0.76, 19.34},
        {0.80, 20.53},  {0.84, 21.51},  {0.88, 22.33},  {0.92, 23.03},  {0.96, 23.81},  {1.00, 24.39},  {1.04, 24.89},
        {1.08, 25.30},  {1.12, 25.66},  {1.16, 25.95},  {1.20, 26.21},  {1.24, 26.42},  {1.28, 26.59},  {1.32, 26.74},
        {1.36, 26.86},  {1.40, 26.96},  {1.44, 27.04},  {1.48, 27.11},  {1.52, 27.17},  {1.56, 27.21},  {1.64, 27.28},
        {1.68, 27.30},  {1.72, 27.32},  {1.76, 27.33},  {1.80, 27.34},  {1.84, 27.35},  {1.88, 27.36},  {1.92, 27.36},
        {1.96, 27.36},  {2.00, 27.36},  {2.10, 27.37},  {2.20, 27.37},  {2.30, 27.38},  {2.40, 27.38},  {2.50, 27.40},
        {2.60, 27.41},  {2.70, 27.43},  {2.80, 27.46},  {2.90, 27.49},  {3.0, 27.52},   {4.0, 27.97},   {5.0, 28.48},
        {6.0, 28.92},   {7.0, 29.28},   {8.0, 29.55},   {9.0, 29.75},   {10.0, 29.89},  {12.0, 30.04},  {14.0, 30.10},
        {16.0, 30.09},  {18.0, 30.05},  {20.0, 30.00},  {25.0, 29.88},  {30.0, 29.80},  {40.0, 29.78},  {50.0, 29.91},
        {60.0, 30.13},  {70.0, 30.38},  {80.0, 30.66},  {90.0, 30.93},  {100.0, 31.19}, {125.0, 31.78}, {150.0, 32.23},
        {175.0, 32.56}, {200.0, 32.79}, {225.0, 32.92}, {250.0, 32.97}, {275.0, 32.96}, {300.0, 32.89}, {325.0, 32.78},
        {350.0, 32.63}, {375.0, 32.44}, {400.0, 32.24}, {425.0, 32.01},
    };
    auto next_val = bystricky.lower_bound(en_at_rest / gconst::kGeV);
    if (next_val == bystricky.begin()) {
      ++next_val;
    }
    const auto tab_val = std::prev(next_val);
    const double slope = (next_val->second - tab_val->second) / (next_val->first - tab_val->first);
    const double intercept = next_val->second - slope * next_val->first;
    x_sect_nn_ = slope * en_at_rest / gconst::kGeV + intercept;
  } else {
    const double mandelstam_s = s_nn_ * s_nn_;
    x_sect_nn_ = 25.0 + 0.146 * std::pow(std::log(mandelstam_s / (gconst::kGeV * gconst::kGeV)), 2);
  }
  generator_ = std::make_unique<TGlauberMC>(name_a.c_str(), name_b.c_str(), x_sect_nn_, -1, time(nullptr));
}
