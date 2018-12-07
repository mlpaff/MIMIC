library(tidyverse)
library(cowplot)

data <- read.csv('./category_means.csv', header=TRUE)
cbPalette <- c('#2AD7F4', '#FFA466')

data$output_label <- factor(data$output_label,
                            levels = c(0, 1),
                            labels = c('No Readmission', 'Readmitted')
                            )

data$cat_label <- factor(data$category,
                         levels = c('total_prior_admits', 'age', 'length_of_stay', 'num_medications',
                                    'num_lab_tests', 'num_diagnosis'),
                         labels = c(
                           'total_prior_admits' = 'Prior Admissions',
                           'age' = 'Age',
                           'length_of_stay' = 'Length of Stay',
                           'num_medications' = 'Prescription Count',
                           'num_lab_test' = 'Number of Tests',
                           'num_diagnosis' = 'Diagnosis Count'
                          )
                         )



plot <- data %>% 
  ggplot(aes(x=output_label, y=value, fill = output_label)) +
  geom_bar(position = 'dodge', stat = 'identity') + 
  scale_y_continuous(expand = c(0, 0, 0.05, 0)) +
  scale_fill_manual(values = cbPalette) +
  facet_wrap(~cat_label, ncol = 3, scales = 'free') +
  panel_border(colour = 'White') +
  theme(
        axis.line.x = element_line(color = 'white'),
        axis.line.y = element_line(color = 'white'),
        axis.ticks = element_line(color = 'white'),
        axis.text = element_text(color = 'white', size = 14),
        strip.text = element_text(margin = margin(2,0,2,0, "mm"), color = 'White', size = 18),
        strip.background = element_rect(fill = NA, color='white', linetype=1, size=0.5),
        axis.title = element_blank(),
        # legend.title = element_blank(),
        legend.position = 'none',
        panel.background = element_rect(fill = "transparent",colour = NA),
        plot.background = element_rect(fill = "transparent",colour = NA)
        # plot.background =  element_rect(fill = '#1B0066') #
        )

plot

# save_plot("./mimic_fig.jpg", plot, base_height=8, base_width=12)

ggsave('eda.png', plot, width=12, height=8, bg='transparent')


feature_labels <- c(
  'total_prior_admits' = 'Prior Admissions',
  'age' = 'Age',
  'gender' = 'Gender',
  'admission_type' = 'Admission Type',
  'perc_tests_abnormal' = 'Abnormal Tests',
  'length_of_stay' = 'Length of Stay',
  'num_medications' = 'Prescription Count',
  'num_lab_tests' = 'Number of Tests',
  'num_diagnosis' = 'Diagnosis Count'
)


features <- read.csv('./feature_importance.csv', header=T)

feature_plot <- features %>% 
  ggplot(aes(x = reorder(Feature, Importance), y = Importance)) + 
  geom_bar(stat='identity', fill='#FFA466') +
  scale_y_continuous(expand = c(0, 0, 0.05, 0)) +
  scale_x_discrete(labels = feature_labels) +
  coord_flip() +
  theme(
    axis.line.x = element_line(color = 'white'),
    axis.line.y = element_line(color = 'white'),
    axis.ticks = element_line(color = 'white'),
    axis.text = element_text(color = 'white', size = 14),
    strip.text = element_text(margin = margin(2,0,2,0, "mm"), color = 'White', size = 16),
    strip.background = element_rect(fill = NA, color='white', linetype=1, size=0.5),
    axis.title = element_blank(),
    # legend.title = element_blank(),
    legend.position = 'none',
    panel.background = element_rect(fill = "transparent",colour = NA),
    plot.background = element_rect(fill = "transparent",colour = NA)
    # plot.background =  element_rect(fill = '#1B0066') #
  )

feature_plot

ggsave('feature_importances.png', feature_plot, width=10, height=6, bg='transparent')


### Plot ROC-AUC curve plots
struct <- read.csv('./data/model1_roc_auc.csv', header=T)
notes <- read.csv('./data/note_model_roc_auc.csv', header=T)
nlp <- read.csv('./data/nlp_model_roc_auc.csv', header=T)

struct <- struct %>% mutate(model = 'Initial')
notes <- notes %>% mutate(model = 'Text')
nlp <- nlp %>% mutate(model = 'NLP')

data <- bind_rows(struct, notes, nlp)

data$model <- factor(data$model,
                     levels = c('Initial', 'Text', 'NLP'))

model_labs <- c(
  'Initial' = 'Chart data: 0.636 auc',
  'Text' = 'NLP: 0.672 auc',
  'NLP' = 'Chart + NLP: 0.693 auc'
)

data %>% select(roc_auc, model) %>% unique()

model1auc <- data %>% filter(model == 'Initial') %>%
  ggplot(aes(x=fpr, y=tpr, fill = model)) +
  geom_line(aes(color = model), size=1.45) +
  geom_abline(color='white', linetype=2, size = 1) + 
  scale_color_manual(values=c('#FF6CB2', '#E8C105', '#2AD7F4'),
                     labels = model_labs) + 
  scale_fill_manual(values=c('#FF6CB2', '#E8C105', '#2AD7F4'),
                    labels = model_labs) +
  scale_x_continuous(expand = c(0, 0), limits=c(0,1.015)) +
  scale_y_continuous(expand = c(0, 0), limits=c(0,1.015)) +
  labs(color = 'Model Accuracy') +
  guides(color = guide_legend(override.aes = list(size = 4))) +
  theme(legend.position = c(0.45, 0.2),
        legend.text = element_text(color = 'white', size=15),
        legend.title = element_text(color = 'White', size = 18),
        legend.key.size = unit(25, 'pt'),
        axis.line.x = element_line(color = 'white'),
        axis.line.y = element_line(color = 'white'),
        axis.ticks = element_line(color = 'white'),
        axis.title = element_text(color='white', size=20),
        axis.text = element_text(color = 'white', size = 14),
        panel.background = element_rect(fill = "transparent",colour = NA),
        plot.background = element_rect(fill = "transparent",colour = NA)
        # plot.background =  element_rect(fill = '#1B0066')
        )

ggsave('model1auc.png', model1auc, width=6, height=6, bg='transparent')

model2auc <- data %>%
  ggplot(aes(x=fpr, y=tpr, fill = model)) +
  geom_line(aes(color = model), size=1.45) +
  geom_abline(color='white', linetype=2, size = 1) + 
  scale_color_manual(values=c('#FF6CB2', '#E8C105', '#2AD7F4'),
                     labels = model_labs) + 
  scale_fill_manual(values=c('#FF6CB2', '#E8C105', '#2AD7F4'),
                     labels = model_labs) + 
  scale_x_continuous(expand = c(0, 0), limits=c(0,1.015)) +
  scale_y_continuous(expand = c(0, 0), limits=c(0,1.015)) +
  labs(color = 'Model Accuracy') +
  guides(color = guide_legend(override.aes = list(size = 4))) +
  theme(legend.position = c(0.45, 0.2),
        legend.text = element_text(color = 'white', size=15),
        legend.title = element_text(color = 'White', size = 18),
        legend.key.size = unit(25, 'pt'),
        axis.line.x = element_line(color = 'white'),
        axis.line.y = element_line(color = 'white'),
        axis.ticks = element_line(color = 'white'),
        axis.title = element_text(color='white', size=20),
        axis.text = element_text(color = 'white', size = 14),
        panel.background = element_rect(fill = "transparent",colour = NA),
        plot.background = element_rect(fill = "transparent",colour = NA)
        # plot.background =  element_rect(fill = '#1B0066')
  )


ggsave('model2auc.png', model2auc, width=6, height=6, bg='transparent')
