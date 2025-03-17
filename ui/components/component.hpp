#ifndef RL_UI_COMPONENTS_COMPONENT
#define RL_UI_COMPONENTS_COMPONENT



namespace rl::ui
{
class IComponent
{
public:
    virtual ~IComponent();
    virtual void draw() = 0;
    virtual void handle_events() = 0;
};
} // namespace rl::ui



#endif
