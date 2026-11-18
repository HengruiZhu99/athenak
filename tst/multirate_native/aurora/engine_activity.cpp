// Read Level Zero Sysman counters once per node, outside the MPI evolution.
#include <level_zero/zes_api.h>

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

namespace {
void Check(ze_result_t result,const char *operation) {
  if (result!=ZE_RESULT_SUCCESS) throw std::runtime_error(
      std::string(operation)+" result="+std::to_string(static_cast<unsigned>(result)));
}
template<typename Handle,typename Query>
std::vector<Handle> Handles(Query query) {
  uint32_t count=0;
  Check(query(&count,nullptr),"enumerate size");
  std::vector<Handle> items(count);
  if (count) Check(query(&count,items.data()),"enumerate handles");
  items.resize(count);
  return items;
}
}

int main() {
  char host[256]{};
  gethostname(host,sizeof(host)-1);
  try {
    Check(zesInit(0),"zesInit");
    for (auto driver:Handles<zes_driver_handle_t>(zesDriverGet)) {
      auto devices=Handles<zes_device_handle_t>([&](auto *count,zes_device_handle_t *items){
        return zesDeviceGet(driver,count,items);
      });
      for (auto device:devices) {
        zes_pci_properties_t pci{};
        pci.stype=ZES_STRUCTURE_TYPE_PCI_PROPERTIES;
        Check(zesDevicePciGetProperties(device,&pci),"PCI address");
        auto engines=Handles<zes_engine_handle_t>([&](auto *count,zes_engine_handle_t *items){
          return zesDeviceEnumEngineGroups(device,count,items);
        });
        for (auto engine:engines) {
          zes_engine_properties_t p{};
          p.stype=ZES_STRUCTURE_TYPE_ENGINE_PROPERTIES;
          Check(zesEngineGetProperties(engine,&p),"engine properties");
          if (p.type!=ZES_ENGINE_GROUP_ALL && p.type!=ZES_ENGINE_GROUP_COMPUTE_ALL) continue;
          zes_engine_stats_t s{};
          auto status=zesEngineGetActivity(engine,&s);
          if (status!=ZE_RESULT_SUCCESS) {
            std::fprintf(stderr,"%s engine %u counter unavailable result=%u\n",host,p.type,status);
            continue;
          }
          std::printf("%s %04x:%02x:%02x.%x %d %u %llu %llu\n",host,pci.address.domain,
              pci.address.bus,pci.address.device,pci.address.function,
              p.onSubdevice?static_cast<int>(p.subdeviceId):-1,static_cast<unsigned>(p.type),
              static_cast<unsigned long long>(s.activeTime),static_cast<unsigned long long>(s.timestamp));
        }
      }
    }
  } catch(const std::exception &error) {
    std::fprintf(stderr,"%s Sysman counters unavailable: %s\n",host,error.what());
    return 1;
  }
}
