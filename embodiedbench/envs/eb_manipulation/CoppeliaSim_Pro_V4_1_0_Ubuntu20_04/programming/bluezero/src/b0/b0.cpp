#include <b0/b0.h>
#include <b0/utils/env.h>
#include <b0/node.h>
#include <b0/logger/logger.h>

#include <iostream>
#include <string>
#include <sstream>
#include <map>

#include <boost/format.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/algorithm/string_regex.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#ifdef HAVE_POSIX_SIGNALS
#include <signal.h>
#endif

namespace b0
{

static void signalHandler(int sig)
{
#ifdef HAVE_POSIX_SIGNALS
    if(sig == SIGINT)  std::cout << " (SIGINT)" << std::endl;
    if(sig == SIGTERM) std::cout << " (SIGTERM)" << std::endl;
#endif
    quit();
}

static void setupSignalHandler()
{
#ifdef HAVE_POSIX_SIGNALS
    struct sigaction sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.sa_handler = signalHandler;
    sigfillset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
#endif
}

namespace po = boost::program_options;

struct Global::Private
{
    bool initialized_{false};
    std::map<std::string, std::string> remap_node_;
    std::map<std::string, std::string> remap_topic_;
    std::map<std::string, std::string> remap_service_;
    logger::Level console_log_level_{logger::Level::info};
    boost::program_options::options_description options_description_{"Allowed options"};
    boost::program_options::positional_options_description positional_options_description_;
    boost::program_options::variables_map variables_map_;
    std::atomic<bool> quit_flag_{false};
    double spin_rate_{10.0};

    void init(Global &g, po::command_line_parser &parser, const std::string &argv0)
    {
        if(initialized_)
            throw std::runtime_error("already initialized");

        setupSignalHandler();

        // process environment variables:
        std::string console_loglevel = b0::env::get("B0_CONSOLE_LOGLEVEL");
        if(console_loglevel != "")
        {
            console_log_level_ = logger::levelInfo(console_loglevel).level;
        }

        // process arguments:
        using str_vec = std::vector<std::string>;
        options_description_.add_options()
            ("help,h", "display help message")
            ("remap,R", po::value<str_vec>()->value_name("oldName=newName")->multitoken()->notifier(boost::bind(&Global::addRemapings, &g, _1)), "remap any name")
            ("remap-node,N", po::value<str_vec>()->value_name("oldName=newName")->multitoken()->notifier(boost::bind(&Global::addNodeRemapings, &g, _1)), "remap a node name")
            ("remap-topic,T", po::value<str_vec>()->value_name("oldName=newName")->multitoken()->notifier(boost::bind(&Global::addTopicRemapings, &g, _1)), "remap a topic name")
            ("remap-service,S", po::value<str_vec>()->value_name("oldName=newName")->multitoken()->notifier(boost::bind(&Global::addServiceRemapings, &g, _1)), "remap a service name")
            ("console-loglevel,L", po::value<std::string>()->default_value(logger::levelInfo(console_log_level_).str), "specify the console loglevel")
            ("spin-rate,F", po::value<double>()->default_value(spin_rate_), "specify the default spin rate")
        ;
        try
        {
            parser.options(options_description_).positional(positional_options_description_);
            po::parsed_options parsed_options = parser.run();
            po::store(parsed_options, variables_map_);
            po::notify(variables_map_);
        }
        catch(po::error &ex)
        {
            std::cerr << "error: " << ex.what() << std::endl;
            g.printUsage(argv0, true);
            std::exit(1);
        }

        if(variables_map_.count("help"))
        {
            g.printUsage(argv0);
            std::exit(0);
        }

        if(variables_map_.count("console-loglevel"))
        {
            console_log_level_ = logger::levelInfo(variables_map_["console-loglevel"].as<std::string>()).level;
        }

        if(variables_map_.count("spin-rate"))
        {
            spin_rate_ = variables_map_["spin-rate"].as<double>();
        }

        initialized_ = true;
    }
};

Global::Global()
    : private_(new Private)
{
}

Global & Global::getInstance()
{
    static Global *global = new Global;
    return *global;
}

static std::vector<std::string> splitAssignment(const std::string &raw_arg)
{
    std::vector<std::string> ret;
    // split by a '=' not preceded by '\':
    boost::split_regex(ret, raw_arg, boost::regex("(?<!\\\\)="));
    if(ret.size() != 2)
        throw std::runtime_error("argument must be origName=newName");
    // process escapes:
    for(std::string &s : ret)
    {
        boost::replace_all(s, "\\=", "=");
        boost::replace_all(s, "\\\\", "\\");
    }
    return ret;
}

void Global::addRemapings(const std::vector<std::string> &raw_arg)
{
    for(auto &s : raw_arg)
    {
        auto x = splitAssignment(s);
        addRemaping(x[0], x[1]);
    }
}

void Global::addNodeRemapings(const std::vector<std::string> &raw_arg)
{
    for(auto &s : raw_arg)
    {
        auto x = splitAssignment(s);
        addNodeRemaping(x[0], x[1]);
    }
}

void Global::addTopicRemapings(const std::vector<std::string> &raw_arg)
{
    for(auto &s : raw_arg)
    {
        auto x = splitAssignment(s);
        addTopicRemaping(x[0], x[1]);
    }
}

void Global::addServiceRemapings(const std::vector<std::string> &raw_arg)
{
    for(auto &s : raw_arg)
    {
        auto x = splitAssignment(s);
        addServiceRemaping(x[0], x[1]);
    }
}

void Global::addRemaping(const std::string &orig_name, const std::string &new_name)
{
    addNodeRemaping(orig_name, new_name);
    addTopicRemaping(orig_name, new_name);
    addServiceRemaping(orig_name, new_name);
}

void Global::addNodeRemaping(const std::string &orig_name, const std::string &new_name)
{
    private_->remap_node_[orig_name] = new_name;
}

void Global::addTopicRemaping(const std::string &orig_name, const std::string &new_name)
{
    private_->remap_topic_[orig_name] = new_name;
}

void Global::addServiceRemaping(const std::string &orig_name, const std::string &new_name)
{
    private_->remap_service_[orig_name] = new_name;
}

void Global::printUsage(const std::string &argv0, bool toStdErr)
{
    std::ostream &os = toStdErr ? std::cerr : std::cout;

    os << "Usage:" << std::endl;

    boost::filesystem::path p(argv0);
    os << "  " << p.filename().string();

    os << " [options]";

    std::string last = "";
    int rep = 0;
    for(int i = 0; i < private_->positional_options_description_.max_total_count(); i++)
    {
        const std::string &n = private_->positional_options_description_.name_for_position(i);
        if(n == last)
        {
            if(!rep) os << " ...";
            if(rep++ > 1000) break;
        }
        else
        {
            os << " " << n;
            last = n;
            rep = 0;
        }
    }
    os << std::endl << std::endl;

    os << private_->options_description_ << std::endl;
}

void Global::addOption(const std::string &name, const std::string &description)
{
    private_->options_description_.add_options()(name.c_str(), description.c_str());
}

void Global::addOptionString(const std::string &name, const std::string &description, std::string *ptr, bool required, const std::string &default_value)
{
    auto value = boost::program_options::value<std::string>(ptr);
    if(required) value->required();
    else value->default_value(ptr ? *ptr : default_value);
    private_->options_description_.add_options()(name.c_str(), value, description.c_str());
}

void Global::addOptionInt(const std::string &name, const std::string &description, int *ptr, bool required, int default_value)
{
    auto value = boost::program_options::value<int>(ptr);
    if(required) value->required();
    else value->default_value(ptr ? *ptr : default_value);
    private_->options_description_.add_options()(name.c_str(), value, description.c_str());
}

void Global::addOptionInt64(const std::string &name, const std::string &description, int64_t *ptr, bool required, int64_t default_value)
{
    auto value = boost::program_options::value<int64_t>(ptr);
    if(required) value->required();
    else value->default_value(ptr ? *ptr : default_value);
    private_->options_description_.add_options()(name.c_str(), value, description.c_str());
}

void Global::addOptionDouble(const std::string &name, const std::string &description, double *ptr, bool required, double default_value)
{
    auto value = boost::program_options::value<double>(ptr);
    if(required) value->required();
    else value->default_value(ptr ? *ptr : default_value);
    private_->options_description_.add_options()(name.c_str(), value, description.c_str());
}

void Global::addOptionStringVector(const std::string &name, const std::string &description, std::vector<std::string> *ptr, bool required, const std::vector<std::string> &default_value)
{
    auto value = boost::program_options::value<std::vector<std::string> >(ptr);
    if(required) value->required();
    private_->options_description_.add_options()(name.c_str(), value, description.c_str());
}

void Global::addOptionIntVector(const std::string &name, const std::string &description, std::vector<int> *ptr, bool required, const std::vector<int> &default_value)
{
    auto value = boost::program_options::value<std::vector<int> >(ptr);
    if(required) value->required();
    private_->options_description_.add_options()(name.c_str(), value, description.c_str());
}

void Global::addOptionInt64Vector(const std::string &name, const std::string &description, std::vector<int64_t> *ptr, bool required, const std::vector<int64_t> &default_value)
{
    auto value = boost::program_options::value<std::vector<int64_t> >(ptr);
    if(required) value->required();
    private_->options_description_.add_options()(name.c_str(), value, description.c_str());
}

void Global::addOptionDoubleVector(const std::string &name, const std::string &description, std::vector<double> *ptr, bool required, const std::vector<double> &default_value)
{
    auto value = boost::program_options::value<std::vector<double> >(ptr);
    if(required) value->required();
    private_->options_description_.add_options()(name.c_str(), value, description.c_str());
}

void Global::setPositionalOption(const std::string &option, int max_count)
{
    private_->positional_options_description_.add(option.c_str(), max_count);
}

int Global::hasOption(const std::string &option)
{
    return private_->variables_map_.count(option);
}

std::string Global::getOptionString(const std::string &option)
{
    return private_->variables_map_[option].as<std::string>();
}

int Global::getOptionInt(const std::string &option)
{
    return private_->variables_map_[option].as<int>();
}

int64_t Global::getOptionInt64(const std::string &option)
{
    return private_->variables_map_[option].as<int64_t>();
}

double Global::getOptionDouble(const std::string &option)
{
    return private_->variables_map_[option].as<double>();
}

std::vector<std::string> Global::getOptionStringVector(const std::string &option)
{
    return private_->variables_map_[option].as<std::vector<std::string> >();
}

std::vector<int> Global::getOptionIntVector(const std::string &option)
{
    return private_->variables_map_[option].as<std::vector<int> >();
}

std::vector<int64_t> Global::getOptionInt64Vector(const std::string &option)
{
    return private_->variables_map_[option].as<std::vector<int64_t> >();
}

std::vector<double> Global::getOptionDoubleVector(const std::string &option)
{
    return private_->variables_map_[option].as<std::vector<double> >();
}

void Global::init(const std::vector<std::string> &argv)
{
    po::command_line_parser parser(argv);
    private_->init(*this, parser, argv.size() ? argv[0] : "?");
}

void Global::init(int &argc, char **argv)
{
    po::command_line_parser parser(argc, argv);
    private_->init(*this, parser, argc ? argv[0] : "?");
}

bool Global::isInitialized() const
{
    return private_->initialized_;
}

static bool makeSubstitutions(const b0::Node &node, std::string &name)
{
    bool ret = false;

    if(name.find("%h") != std::string::npos)
    {
        boost::replace_all(name, "%h", node.hostname());
        ret = true;
    }

    if(name.find("%n") != std::string::npos)
    {
        boost::replace_all(name, "%n", node.getName());
        ret = true;
    }

    return ret;
}

static bool remapName(const b0::Node &node, const std::map<std::string, std::string> &map, const std::string &name, std::string &remapped_name)
{
    auto i = map.find(name);
    bool ret = false;
    if(i == map.end())
    {
        remapped_name = name;
    }
    else
    {
        remapped_name = i->second;
        ret = true;
    }
    return makeSubstitutions(node, remapped_name) || ret;
}

static std::string getRemappedName(const b0::Node &node, const std::map<std::string, std::string> &map, const std::string &name)
{
    std::string ret = name;
    remapName(node, map, name, ret);
    return ret;
}

std::string Global::getRemappedNodeName(const b0::Node &node, const std::string &node_name)
{
    return getRemappedName(node, private_->remap_node_, node_name);
}

std::string Global::getRemappedTopicName(const b0::Node &node, const std::string &topic_name)
{
    return getRemappedName(node, private_->remap_topic_, topic_name);
}

std::string Global::getRemappedServiceName(const b0::Node &node, const std::string &service_name)
{
    return getRemappedName(node, private_->remap_service_, service_name);
}

bool Global::remapNodeName(const b0::Node &node, const std::string &node_name, std::string &remapped_node_name)
{
    return remapName(node, private_->remap_node_, node_name, remapped_node_name);
}

bool Global::remapTopicName(const b0::Node &node, const std::string &topic_name, std::string &remapped_topic_name)
{
    return remapName(node, private_->remap_topic_, topic_name, remapped_topic_name);
}

bool Global::remapServiceName(const b0::Node &node, const std::string &service_name, std::string &remapped_service_name)
{
    return remapName(node, private_->remap_service_, service_name, remapped_service_name);
}

logger::Level Global::getConsoleLogLevel()
{
    return private_->console_log_level_;
}

void Global::setConsoleLogLevel(logger::Level level)
{
    private_->console_log_level_ = level;
}

double Global::getSpinRate()
{
    return private_->spin_rate_;
}

void Global::setSpinRate(double rate)
{
    if(rate <= 0)
        throw std::range_error("Spin rate must be positive");
    private_->spin_rate_ = rate;
}

bool Global::quitRequested()
{
    return private_->quit_flag_.load();
}

void Global::quit()
{
    private_->quit_flag_.store(true);
}

void init(const std::vector<std::string> &argv)
{
    try
    {
        Global::getInstance().init(argv);
    }
    catch(std::exception &ex)
    {
        std::cerr << "Initialization failed: " << ex.what() << std::endl;
        std::exit(100);
    }
}

void init(int &argc, char **argv)
{
    try
    {
        Global::getInstance().init(argc, argv);
    }
    catch(std::exception &ex)
    {
        std::cerr << "Initialization failed: " << ex.what() << std::endl;
        std::exit(100);
    }
}

bool isInitialized()
{
    return Global::getInstance().isInitialized();
}

void printUsage(const std::string &argv0, bool toStdErr)
{
    Global::getInstance().printUsage(argv0, toStdErr);
}

void addOption(const std::string &name, const std::string &description)
{
    Global::getInstance().addOption(name, description);
}

void addOptionString(const std::string &name, const std::string &description, std::string *ptr, bool required, const std::string &default_value)
{
    Global::getInstance().addOptionString(name, description, ptr, required, default_value);
}

void addOptionInt(const std::string &name, const std::string &description, int *ptr, bool required, int default_value)
{
    Global::getInstance().addOptionInt(name, description, ptr, required, default_value);
}

void addOptionInt64(const std::string &name, const std::string &description, int64_t *ptr, bool required, int64_t default_value)
{
    Global::getInstance().addOptionInt64(name, description, ptr, required, default_value);
}

void addOptionDouble(const std::string &name, const std::string &description, double *ptr, bool required, double default_value)
{
    Global::getInstance().addOptionDouble(name, description, ptr, required, default_value);
}

void addOptionStringVector(const std::string &name, const std::string &description, std::vector<std::string> *ptr, bool required, const std::vector<std::string> &default_value)
{
    Global::getInstance().addOptionStringVector(name, description, ptr, required, default_value);
}

void addOptionIntVector(const std::string &name, const std::string &description, std::vector<int> *ptr, bool required, const std::vector<int> &default_value)
{
    Global::getInstance().addOptionIntVector(name, description, ptr, required, default_value);
}

void addOptionInt64Vector(const std::string &name, const std::string &description, std::vector<int64_t> *ptr, bool required, const std::vector<int64_t> &default_value)
{
    Global::getInstance().addOptionInt64Vector(name, description, ptr, required, default_value);
}

void addOptionDoubleVector(const std::string &name, const std::string &description, std::vector<double> *ptr, bool required, const std::vector<double> &default_value)
{
    Global::getInstance().addOptionDoubleVector(name, description, ptr, required, default_value);
}

void setPositionalOption(const std::string &option, int max_count)
{
    Global::getInstance().setPositionalOption(option, max_count);
}

int hasOption(const std::string &option)
{
    return Global::getInstance().hasOption(option);
}

std::string getOptionString(const std::string &option)
{
    return Global::getInstance().getOptionString(option);
}

int getOptionInt(const std::string &option)
{
    return Global::getInstance().getOptionInt(option);
}

int64_t getOptionInt64(const std::string &option)
{
    return Global::getInstance().getOptionInt64(option);
}

double getOptionDouble(const std::string &option)
{
    return Global::getInstance().getOptionDouble(option);
}

std::vector<std::string> getOptionStringVector(const std::string &option)
{
    return Global::getInstance().getOptionStringVector(option);
}

std::vector<int> getOptionIntVector(const std::string &option)
{
    return Global::getInstance().getOptionIntVector(option);
}

std::vector<int64_t> getOptionInt64Vector(const std::string &option)
{
    return Global::getInstance().getOptionInt64Vector(option);
}

std::vector<double> getOptionDoubleVector(const std::string &option)
{
    return Global::getInstance().getOptionDoubleVector(option);
}

logger::Level getConsoleLogLevel()
{
    return Global::getInstance().getConsoleLogLevel();
}

void setConsoleLogLevel(logger::Level level)
{
    Global::getInstance().setConsoleLogLevel(level);
}

double getSpinRate()
{
    return Global::getInstance().getSpinRate();
}

void setSpinRate(double rate)
{
    Global::getInstance().setSpinRate(rate);
}

bool quitRequested()
{
    return Global::getInstance().quitRequested();
}

void quit()
{
    Global::getInstance().quit();
}

using boost::program_options::value;

} // namespace b0

