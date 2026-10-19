#include <CGlauberFactory.hh>
#include <COLA.hh>
#include <GlauberConst.hh>
#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

extern "C" cola::VModule* LoadCOLAModule();

namespace {

  class TestWriter final : public cola::VWriter {
   public:
    void operator()(std::unique_ptr<cola::EventData>&& /*data*/) override {}
  };

  class TestWriterFactory final : public cola::VWriterFactory {
   public:
    std::unique_ptr<cola::VFilter> Create(const std::unordered_map<std::string, std::string>& /*meta_data*/) override {
      return std::make_unique<TestWriter>();
    }

    const std::string& GetFilterName() const override {
      static const std::string name{"test_writer"};
      return name;
    }
  };

  std::unordered_map<std::string, std::string> DefaultGlauberParams() {
    return {
        {"name_a", "Au"}, {"name_b", "Au"}, {"energy", "200"}, {"is_collider", "1"}, {"fermiMom", "1"},
    };
  }

  void ExpectCrossSection(const std::string& energy, double expected, bool is_collider = false) {
    SCOPED_TRACE(energy);
    auto params = DefaultGlauberParams();
    params["energy"] = energy;
    params["is_collider"] = is_collider ? "1" : "0";
    CGlauberFactory factory;
    const auto filter = factory.Create(params);
    auto* generator = dynamic_cast<cola::VGenerator*>(filter.get());
    ASSERT_NE(generator, nullptr);
    const auto event = (*generator)();
    ASSERT_NE(event, nullptr);
    EXPECT_NEAR(event->ini_state.sect_nn, expected, 1.e-5);
  }

}  // namespace

TEST(CGlauberTest, ModuleExposesGeneratorFactory) {
  const auto module = std::unique_ptr<cola::VModule>(LoadCOLAModule());
  const cola::FactoryMap factories = module->GetModuleFilters();

  ASSERT_NE(factories.find("CGlauber"), factories.end());
  EXPECT_EQ(factories.at("CGlauber")->GetFilterType(), cola::FilterType::kGenerator);
  EXPECT_EQ(factories.at("CGlauber")->GetFilterName(), "CGlauber");
}

TEST(CGlauberTest, FactoryCreatesGenerator) {
  CGlauberFactory factory;
  const auto filter = factory.Create(DefaultGlauberParams());
  ASSERT_NE(filter, nullptr);
  EXPECT_EQ(factory.GetFilterName(), "CGlauber");
}

TEST(CGlauberTest, GeneratorProducesNonEmptyEvent) {
  CGlauberFactory factory;
  const auto filter = factory.Create(DefaultGlauberParams());
  auto* generator = dynamic_cast<cola::VGenerator*>(filter.get());
  ASSERT_NE(generator, nullptr);

  const auto event = (*generator)();
  ASSERT_NE(event, nullptr);
  EXPECT_FALSE(event->particles.empty());
  EXPECT_FALSE(event->ini_state.ini_state_particles.empty());
}

TEST(CGlauberTest, ParseAndRunViaMetaProcessor) {
  const std::string xml = R"(<?xml version="1.0"?>
<program>
    <generator name="CGlauber" name_a="Au" name_b="Au" energy="200" is_collider="1" fermiMom="1"/>
    <writer name="test_writer"/>
</program>
)";

  const auto module = std::unique_ptr<cola::VModule>(LoadCOLAModule());
  cola::MetaProcessor processor(module->GetModuleFilters());
  processor.Register(std::make_unique<TestWriterFactory>());
  std::istringstream stream(xml);
  cola::FilterEnsemble ensemble = processor.Parse(stream);

  ASSERT_NE(ensemble.generator, nullptr);
  const auto event = (*ensemble.generator)();
  ASSERT_NE(event, nullptr);
  EXPECT_FALSE(event->particles.empty());
}

TEST(CGlauberTest, FixedTargetCrossSectionMatchesTable) {
  for (const auto& [energy, expected] : {std::pair{"0.28", 0.027},
                                         {"0.29", 0.051},
                                         {"0.50", 5.87},
                                         {"1.00", 24.39},
                                         {"20.0", 30.00},
                                         {"200.0", 32.79},
                                         {"400.0", 32.24}}) {
    ExpectCrossSection(energy, expected);
  }
}

TEST(CGlauberTest, FixedTargetCrossSectionInterpolatesBracketingPoints) {
  for (const auto& [energy, expected] : {std::pair{"0.285", 0.039}, {"5.5", 28.70}, {"210.0", 32.842}}) {
    ExpectCrossSection(energy, expected);
  }
}

TEST(CGlauberTest, FixedTargetCrossSectionUsesLastTableInterval) {
  for (const auto& [energy, expected] : {std::pair{"400.01", 32.239908}, {"410.0", 32.148}, {"424.999", 32.0100092}}) {
    ExpectCrossSection(energy, expected);
  }
}

TEST(CGlauberTest, ColliderCrossSectionUsesEquivalentFixedTargetEnergy) {
  const auto mass = gconst::kNucleonAverMass / gconst::kGeV;
  const auto collider_energy = std::sqrt(4 * mass * mass + 2 * mass * 410.0);
  ExpectCrossSection(std::to_string(collider_energy), 32.148, true);
}
